import subprocess
import time
import yaml
from pathlib import Path
import sys
import threading
import logging
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.evaluate import evaluate, detector_ready, wartime_stop_event, peacetime_stop_event, peacetime_done, cfg_safe_get, format_duration
from config.dns_config_generator import generate_all_configs

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DOCKER_PATH = PROJECT_ROOT / "docker"
DOCKER_COMPOSE_FILE = DOCKER_PATH / "docker-compose.yml"
DOCKER_ENV_FILE = DOCKER_PATH / ".env"
CONFIG_TEMPLATE_DIR = PROJECT_ROOT / "config" / "templates"
RESULTS_DIR = PROJECT_ROOT / "results"
DETECTOR_PATH = PROJECT_ROOT / "detection" / "detectors"
LOGFILE = DOCKER_PATH / "dns-collector" / "logs" / "dnslogs.json"

class UnifiedOutputHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.last_progress_length = 0
        self._lock = threading.Lock()
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record):
        try:
            msg = self.format(record)
            is_progress = getattr(record, 'progress', False)

            with self._lock:
                if is_progress:
                    padded_msg = msg.ljust(self.last_progress_length)
                    sys.stdout.write(f"\r{padded_msg}")
                    sys.stdout.flush()
                    self.last_progress_length = len(msg)
                else:
                    if self.last_progress_length > 0:
                        sys.stdout.write("\r" + " " * self.last_progress_length + "\r")
                        self.last_progress_length = 0
                    sys.stdout.write(msg + "\n")
                    sys.stdout.flush()
        except Exception:
            self.handleError(record)

logger = logging.getLogger('dns_detector')
logger.setLevel(logging.INFO)
logger.handlers.clear()

# Console handler with custom formatting
console_handler = UnifiedOutputHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

logger.propagate = False


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def write_docker_env(pcap_dir):
    DOCKER_ENV_FILE.write_text(f"PCAP_PATH={pcap_dir}\n")


def compose_up(profile):
    cmd = [
        "docker",
        "compose",
        "-f", str(DOCKER_COMPOSE_FILE),
        "--profile", profile,
        "up",
        "-d",
    ]
    subprocess.run(cmd, check=True)


def compose_down():
    cmd = [
        "docker",
        "compose",
        "-f", str(DOCKER_COMPOSE_FILE),
        "--profile", "*",
        "down"
    ]
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.warning(f"[!] Docker compose down returned code {result.returncode}")
            if result.stderr:
                logger.warning(f"[!] Error: {result.stderr.strip()}")
        else:
            logger.info("[+] Docker containers stopped successfully")
    except Exception as e:
        logger.error(f"[!] Failed to run docker compose down: {e}")


def docker_exec(container: str, *docker_cmd: str, check: bool = True):
    cmd = ["docker", "exec", container] + list(docker_cmd)
    subprocess.run(cmd, check=check)


def replay_fire_and_forget(pcap: str, pps: int, label: str):
    """Fire and forget: startet tcpreplay ohne zu warten"""
    if not pcap:
        logger.warning(f"[-] Skipping {label} replay (disabled)")
        return
    
    cmd = ["tcpreplay"]
    if pps is None or pps <= 0:
        logger.info(f"[+] Firing {label} replay at original speed from {pcap}")
        cmd.extend(["-i", "replay0", pcap])
    else:
        logger.info(f"[+] Firing {label} replay: {pps} PPS from {pcap}")
        cmd.extend([f"--pps={pps}", "-i", "replay0", pcap])
    
    docker_exec("pcap-replayer", *cmd, check=False)

def main():
    cfg = load_config(CONFIG_PATH)
    duration = int(cfg_safe_get(cfg, ["timing", "duration"], 60)) * 60
    peacetime_duration = cfg_safe_get(cfg, ["timing", "peacetime_duration"]) * 60
    tunneling_domains = cfg_safe_get(cfg, ["traffic", "tunnel", "tunneling_domains"], [])
    server_ip = cfg_safe_get(cfg, ['traffic', 'tunnel', 'tunnel_server_ip'], '192.168.4.4')
    public_resolver = cfg_safe_get(cfg, ['global', 'public_resolver'], '1.1.1.1')
    allowlist = cfg_safe_get(cfg, ['allowlist', 'global_allowlist_path'])
    active_detectors = cfg_safe_get(cfg, ['detectors', 'active'], None)
    

    if not peacetime_duration:
        peacetime_duration = 0

    if allowlist is None:
        allowlist = []
    
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SETUP")
    logger.info("="*60)
    logger.info(f"[+] Experiment: {cfg_safe_get(cfg, ['global', 'name'], '')}")
    logger.info(f"[+] Peacetime Duration: {format_duration(peacetime_duration)}")
    logger.info(f"[+] Wartime Duration: {format_duration(duration - peacetime_duration)}")
    logger.info(f"[+] Total Duration: {format_duration(duration)}")
    logger.info("="*60)

    results_file = RESULTS_DIR / f"results_{cfg_safe_get(cfg, ['global', 'name'], 'experiment')}_{int(time.time())}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    t_detect = None

    try:
        # Ensure docker is stopped before cleaning up
        logger.info("[+] Ensuring clean state...")
        try:
            compose_down()
        except:
            pass  # Ignore errors if nothing is running
        
        # Clear old logfile to start fresh
        LOGFILE.parent.mkdir(parents=True, exist_ok=True)
        if LOGFILE.exists():
            file_size = LOGFILE.stat().st_size / (1024*1024)
            logger.info(f"[+] Clearing old logfile ({file_size:.1f} MB)")
            LOGFILE.unlink()
        LOGFILE.touch()

        write_docker_env(PROJECT_ROOT / cfg_safe_get(cfg, ['traffic', 'pcap_path']))
        generate_all_configs(tunneling_domains, server_ip, public_resolver, CONFIG_TEMPLATE_DIR, DOCKER_PATH)
        logger.info("[+] Starting docker compose")
        compose_up("essential")
        time.sleep(5)

        t_detect = threading.Thread(
            target=evaluate,
            args=(DETECTOR_PATH, str(LOGFILE), tunneling_domains, allowlist, str(results_file), cfg, active_detectors),
        )
        t_detect.start()

        if not detector_ready.wait(timeout=30):
            logger.error("[!] Detector initialization timed out after 30 seconds")
            raise TimeoutError("Detector initialization timed out")
        logger.info("[+] Detection ready")
        end_time = time.time() + duration
        logger.info(f"[+] Estimated end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

        logger.info("\n" + "="*60)
        logger.info("TRAFFIC GENERATION")
        logger.info("="*60)

        #fire-and-forget
        benign_enabled = bool(cfg_safe_get(cfg, ["traffic", "benign", "enabled"], False))
        benign_pcap = cfg_safe_get(cfg, ["traffic", "benign", "pcap"], "")
        benign_pps = cfg_safe_get(cfg, ["traffic", "benign", "pps"])

        wildcard_enabled = bool(cfg_safe_get(cfg, ["traffic", "wildcard", "enabled"], False))
        wildcard_pcap = cfg_safe_get(cfg, ["traffic", "wildcard", "pcap"], "")
        wildcard_pps = cfg_safe_get(cfg, ["traffic", "wildcard", "pps"])

        tunneling_enabled = bool(cfg_safe_get(cfg, ["traffic", "tunnel", "replay"], False))
        tunneling_pcap = cfg_safe_get(cfg, ["traffic", "tunnel", "pcap"], "")
        tunneling_pps = cfg_safe_get(cfg, ["traffic", "tunnel", "pps"])

        # Start benign traffic
        if benign_enabled:
                threading.Thread(
                    target=replay_fire_and_forget,
                    args=(benign_pcap, benign_pps, "benign"),
                    daemon=True, 
                ).start()

        if wildcard_enabled:
            threading.Thread(
                target=replay_fire_and_forget,
                args=(wildcard_pcap, wildcard_pps, "wildcard"),
                daemon=True,  
            ).start()

        if wildcard_enabled or benign_enabled:
            logger.info(f"[+] Benign traffic started - running for {format_duration(duration)}...")
        else:
            logger.info(f"[+] No benign traffic - waiting for external DNS traffic or wartime...")


        if peacetime_duration and peacetime_duration != 0:
            logger.info(f"[+] Peacetime started: Running for {format_duration(peacetime_duration)}...")
            time.sleep(peacetime_duration)
            peacetime_stop_event.set()
            if not peacetime_done.wait(timeout=10):
                logger.warning("[!] Peacetime cleanup timed out, continuing anyway")
            else:
                logger.info(f"[+] Peacetime done! Remainaing duration for wartime: {format_duration(duration - peacetime_duration)}")

        # Start tunnel traffic if enabled
        if tunneling_enabled:
            threading.Thread(
                target=replay_fire_and_forget,
                args=(tunneling_pcap, tunneling_pps, "tunnel"),
                daemon=True,  
            ).start()
            logger.info(f"[+] Tunnel replay started - running for {format_duration(duration - peacetime_duration)}...")

        if cfg_safe_get(cfg, ["traffic", "tunnel", "docker"]):
            toolname = cfg_safe_get(cfg, ["traffic", "tunnel", "toolname"])
            logger.info(f"[+] Starting tunneling tool: {toolname}")
            compose_up(toolname)

        remaining_time = duration - peacetime_duration
        if not wartime_stop_event.wait(timeout=remaining_time):
            logger.info("[+] Experiment duration reached - shutting down")
        else:
            logger.info("[+] Detection completed early - shutting down")
        wartime_stop_event.set()
        t_detect.join()
        compose_down()


    except KeyboardInterrupt:
        logger.info("\n[!] Interrupted by user - shutting down...")
        peacetime_stop_event.set()
        wartime_stop_event.set()
        if t_detect is not None and t_detect.is_alive():
            logger.info("[+] Waiting for detection thread to finish...")
            t_detect.join(timeout=10)
        logger.info("[+] Stopping docker containers...")
        compose_down()
    except Exception as e:
        logger.error(f"[!] Error: {e}", exc_info=True)
        peacetime_stop_event.set()
        wartime_stop_event.set()
        if t_detect is not None and t_detect.is_alive():
            t_detect.join(timeout=5)
        compose_down()
    finally:
        try:
            compose_down()
        except:
            pass


if __name__ == "__main__":
    main()
