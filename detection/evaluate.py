import os
import importlib
import inspect
import threading
import logging
from detection.detector_base.readfile import readjson
from detection.detector_base.feature_extraction import get_registered_domain, is_response, extract_record_type, extract_subdomain_string, json_safe_get, get_fqdn, validate_fqdn
import json
from datetime import datetime

# Get the logger configured in main.py
logger = logging.getLogger('dns_detector')

def cfg_safe_get(cfg: dict, path: list[str], default: any = None) -> any:
    """safe get values from yaml with default values"""
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def format_duration(seconds: int) -> str:
    """Convert seconds to human-readable format (e.g., '1h20m' or '5m30s')"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 and hours == 0:  # Only show seconds if no hours
        parts.append(f"{secs}s")
    
    return "".join(parts) or "0s"

detector_ready = threading.Event()
wartime_stop_event = threading.Event()
peacetime_stop_event = threading.Event()
peacetime_done = threading.Event()

def load_detectors(detectors_path, active_detectors=None):
    """Load detector classes dynamically, filtered by active_detectors list."""
    detectors = []
    for folder in os.listdir(detectors_path):
        folder_path = os.path.join(detectors_path, folder)
        if not os.path.isdir(folder_path) or folder.startswith("__"):
            continue
        # Filter by active_detectors if provided
        if active_detectors is not None and folder not in active_detectors:
            logger.info(f"⊘ Skipping {folder} (not in active detectors)")
            continue
        detector_py_path = os.path.join(folder_path, "detector.py")
        if not os.path.isfile(detector_py_path):
            logger.warning(f"⚠ Skipping {folder} (no detector.py)")
            continue
        logger.info(f"✓ Loading: {folder}")
        try:
            module = importlib.import_module(f"detection.detectors.{folder}.detector")
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module.__name__:
                    detectors.append(obj)
        except ImportError as e:
            logger.error(f"✗ Import error in {folder}: {e}")
    return detectors

def evaluate(detector_path: str, logfile: str, tunneling_domains: list[str],
             allowlist_path: str, results_file: str, cfg: dict,
             active_detectors: list[str] = None) -> dict:

    ### INITILIZATION ###
    detector_classes = load_detectors(detector_path, active_detectors)
    detectors = [cls() for cls in detector_classes]
    logger.info(f"\n{len(detectors)} detectors loaded: {[type(d).__name__ for d in detectors]}\n")
    alldomains = set()
    # This allows up to 60 seconds of no data before considering file complete
    log_generator = readjson(logfile, timeout_seconds=30, max_timeouts=2)

    tunneling_domains = set(tunneling_domains)

    global_allowlist = set()

    if allowlist_path:
        with open(allowlist_path, 'r') as file:
            global_allowlist.update(file.read().splitlines())


    ### PEACETIME ###
    duration = int(cfg_safe_get(cfg, ["timing", "duration"], 60))
    peacetime_duration = cfg_safe_get(cfg, ["timing", "peacetime_duration"], duration // 3)
    if not peacetime_duration:
        peacetime_duration = 0

    detector_ready.set()

    start_timestamp = datetime.now()

    if peacetime_duration != 0:

        logger.info("\n" + "="*60)
        logger.info("PEACETIME PHASE - Building Allowlist")
        logger.info("="*60)

        peace_logcount = 0

        while not peacetime_stop_event.is_set():
            try:
                logline = next(log_generator)
            except StopIteration:
                logger.warning(f"\n⚠ Log ended after {peace_logcount} lines")
                break
            
            # Handle timeout from generator
            if logline is None:
                continue

            domain = get_registered_domain(logline)
            if domain in tunneling_domains:
                logger.warning(f"[!] WARNING! Tunneling domain {domain} seen during peacetime!")

            if not validate_fqdn(logline) or not domain:
                logger.debug(f"[!] Invalid FQDN detected: {get_fqdn(logline)}")
                continue
            
            # Only update progress every 500 lines
            if peace_logcount % 100 == 0:
                logger.info(f"[PEACETIME] Processing line {peace_logcount} ({domain})", extra={'progress': True})
            
            peace_logcount += 1

            for detector in detectors:
                detector.process_line(logline, False)



    # Pfade zu den gespeicherten detektorspezifischen Allowlists
    saved_allowlists_dir = cfg_safe_get(cfg, ["allowlist", "local_allowlist_dir"], None)

    for detector in detectors:
        detector_name = type(detector).__name__

        if peacetime_duration != 0:
            # Frisch aus Peacetime generiert
            local_allowlist = detector.alarms
        elif saved_allowlists_dir:
            # Aus gespeicherter Datei laden
            path = os.path.join(saved_allowlists_dir, f"{detector_name.lower()}.txt")
            local_allowlist = set()
            if os.path.exists(path):
                with open(path, 'r') as f:
                    local_allowlist = set(f.read().splitlines())
                logger.info(f"[+] {detector_name}: loaded saved allowlist from {path}")
            else:
                logger.warning(f"[!] {detector_name}: no saved allowlist found at {path}")
        else:
            local_allowlist = set()

        detector.reset()
        detector.allowlist = local_allowlist.union(global_allowlist)
        logger.info(f"[+] {detector_name} allowlist size: {len(detector.allowlist)}")

    peacetime_done.set()
    
    ### WARTIME ###

    logger.info("\n" + "="*60)
    logger.info("WARTIME PHASE - Detection Active")
    logger.info("="*60)

    tunneling_domains_seen = {domain: False for domain in tunneling_domains}
    
    logcount = 0
    tunnel_count = 0

    while not wartime_stop_event.is_set():
        
        try:
            logline = next(log_generator)
        except StopIteration:
            logger.warning(f"\n⚠ Log ended after {logcount} lines")
            break 

        # Handle timeout from generator
        if logline is None or get_fqdn(logline) is None:
            continue

        domain = get_registered_domain(logline)
        
        if not validate_fqdn(logline) or domain == "":
            logger.debug(f"[!] Invalid FQDN detected: {get_fqdn(logline)}")
            continue

        domain = domain.lower()

        if domain and domain not in alldomains:
            alldomains.add(domain)

        # Only update progress every 500 lines
        if logcount % 100 == 0:
            logger.info(f"[WARTIME] Processing line {logcount} ({domain})", extra={'progress': True})
        
        istunnel = False
        logcount += 1
        
        if domain in tunneling_domains:
            tunneling_domains_seen[domain] = True
            istunnel = True
            tunnel_count += 1
            logger.debug(f"[!] Tunneling seen...({domain})")
        
        for detector in detectors:
            detector.process_line(logline, istunnel)

        done = True

        for detector in detectors:
            if detector.alarms.intersection(tunneling_domains) != tunneling_domains or len(tunneling_domains) == 0:
                done = False
                break

        if done:
            logger.info("[+] All tunneling domains detected. Stopping Wartime...")
            wartime_stop_event.set()
    
    end_timestamp = datetime.now()
    
    # Clear the progress line
    logger.info("")  
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION PHASE - Computing Metrics")
    logger.info("="*60)
    
    seen_tunnels = set([domain for domain, seen in tunneling_domains_seen.items() if seen])
    
    results = []
    for detector in detectors:
        logger.info(f"\n=== Evaluation of {type(detector).__name__} ===")
        results.append(detector.evaluate(seen_tunnels, alldomains, logcount, tunnel_count, global_allowlist))
        logger.info("")
    
    # Build comprehensive metadata
    metadata = {
        "experiment": {
            "name": cfg_safe_get(cfg, ['global', 'name'], 'unknown'),
            "description": cfg_safe_get(cfg, ['global', 'description']),
            "start_timestamp": start_timestamp.isoformat(),
            "end_timestamp": end_timestamp.isoformat(),
            "duration": format_duration(cfg_safe_get(cfg, ['timing', 'duration'], 0) * 60),
            "actual_runtime": format_duration((end_timestamp - start_timestamp).total_seconds()),
        },
        "traffic_config": {
            "benign": {
                "enabled": cfg_safe_get(cfg, ['traffic', 'benign', 'enabled'], False),
                "pcap": cfg_safe_get(cfg, ['traffic', 'benign', 'pcap'], ''),
                "pps": cfg_safe_get(cfg, ['traffic', 'benign', 'pps'], 0),
                "loop": cfg_safe_get(cfg, ['traffic', 'benign', 'loop'], False)
            },
            "wildcard": {
                "enabled": cfg_safe_get(cfg, ['traffic', 'wildcard', 'enabled'], False),
                "pczap": cfg_safe_get(cfg, ['traffic', 'wildcard', 'pcap'], ''),
                "pps": cfg_safe_get(cfg, ['traffic', 'wildcard', 'pps'], 0),
                "loop": cfg_safe_get(cfg, ['traffic', 'wildcard', 'loop'], False)
            },
            "tunnel": {
                "replay_enabled": cfg_safe_get(cfg, ['traffic', 'tunnel', 'replay'], False),
                "pcap": cfg_safe_get(cfg, ['traffic', 'tunnel', 'pcap'], ''),
                "pps": cfg_safe_get(cfg, ['traffic', 'tunnel', 'pps'], 0),
                "loop": cfg_safe_get(cfg, ['traffic', 'tunnel', 'loop'], False),
                "tunneling_domains": list(tunneling_domains),
                "tunneling_domains_seen": list(seen_tunnels),
                "tunnel_server_ip": cfg_safe_get(cfg, ['traffic', 'tunnel', 'tunnel_server_ip'], ''),
                "expansion_factor": cfg_safe_get(cfg, ['traffic', 'tunnel', 'expansion_factor'], 1)
            },
            "pcap_path": cfg_safe_get(cfg, ['traffic', 'pcap_path'], ''),
            "allowlist_path": cfg_safe_get(cfg, ["allowlist", "local_allowlist_dir"], None)
        },
        "dataset_statistics": {
            "total_loglines": logcount,
            "tunnel_loglines": tunnel_count,
            "tunnel_percentage": (tunnel_count / logcount * 100) if logcount > 0 else 0,
            "unique_domains": len(alldomains),
            "logfile": logfile
        },
        "global_config": {
            "public_resolver": cfg_safe_get(cfg, ['global', 'public_resolver'], ''),
            "detector_path": str(detector_path)
        },
        "complete_config": cfg
    }
    
    # Comprehensive output structure
    output = {
        "metadata": metadata,
        "detectors": results
    }
    
    # Save results to file
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"[+] Results saved to {results_file}")
    
    return output
