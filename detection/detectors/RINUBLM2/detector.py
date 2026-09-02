from detection.detector_base.detector_base import Detector
from detection.detector_base.feature_extraction import get_registered_domain, extract_subdomain_string, extract_record_type, extract_type, json_safe_get, extract_timestamp_date, extract_query_ip, get_txt_content


# DNS Intrusion Detection (DID) — A SNORT-based solution to detect DNS Amplification and DNS Tunneling attacks

class RINUBLM2(Detector):

    def __init__(self):
        super().__init__()
        self.iodine_null_tracker = {}
        self.iodine_cname_tracker = {}
        self.dnscat_plain_tracker = {}
        self.dnscat_cname_tracker = {}
        self.dnscat_mx_tracker = {}
        self.dnscat_txt_tracker = {}
        self.thunderdns_tracker = {}
        self.ozyman_up_tracker = {}
        self.ozyman_down_tracker = {}


    def _is_iodine_packet(self, logline: dict) -> str:
        """Check if the packet matches a Iodine signature."""
        qdcount = int(json_safe_get(logline, 'dns.qdcount'))
        ancount = int(json_safe_get(logline, 'dns.ancount'))
        subdomain = extract_subdomain_string(logline)
        record_type = extract_record_type(logline)
        record_class = extract_type(logline)

        if qdcount == 1 and ancount == 1 and "yrb" in subdomain and record_class == "IN":
            if record_type == "NULL" or record_type == "CNAME":
                return record_type

    def _is_dnscat2_packet(self, logline: dict) -> str:
        """Check if the packet matches a dnscat2 signature."""
        qdcount = int(json_safe_get(logline, 'dns.qdcount'))
        ancount = int(json_safe_get(logline, 'dns.ancount'))
        subdomain = extract_subdomain_string(logline)
        record_type = extract_record_type(logline)
        record_class = extract_type(logline)

        if qdcount == 1 and ancount == 0 and record_class == "IN":
            if subdomain.startswith("dnscat"):
                return "PLAIN"
            if record_type in ["CNAME", "MX", "TXT"]:
                if json_safe_get(logline, 'dns.length') > 100:
                    return record_type

    def _is_dns2tcp_packet(self, logline: dict) -> bool:
        """Check if the packet matches the dns2tcp signature."""
        qdcount = int(json_safe_get(logline, 'dns.qdcount'))
        ancount = int(json_safe_get(logline, 'dns.ancount'))
        subdomain = extract_subdomain_string(logline)
        record_type = extract_record_type(logline)
        record_class = extract_type(logline)

        if qdcount == 1 and ancount == 0 and "=auth" in subdomain and record_type == "TXT" and record_class == "IN":
            return True
        
        return False

    def _is_thunderdns_packet(self, logline: dict) -> bool:
        """Check if the packet matches the ThunderDNS signature."""
        qdcount = int(json_safe_get(logline, 'dns.qdcount'))
        ancount = int(json_safe_get(logline, 'dns.ancount'))
        record_type = extract_record_type(logline)
        record_class = extract_type(logline)

        if qdcount == 1 and ancount == 1 and record_type == "TXT" and record_class == "IN" and "ND" in get_txt_content(logline):
            return True
        
        return False

    def _is_ozymandns_packet(self, logline: dict) -> str:
        """Check if the packet matches the OzymanDNS signature."""
        qdcount = int(json_safe_get(logline, 'dns.qdcount'))
        ancount = int(json_safe_get(logline, 'dns.ancount'))
        subdomain = extract_subdomain_string(logline)
        record_type = extract_record_type(logline)
        record_class = extract_type(logline)

        if qdcount == 1 and ancount == 0 and record_class == "IN":
            if "id-" in subdomain and "down" in subdomain and record_type == "TXT":
                return "DOWN"
            if "-0" in subdomain and "id-" in subdomain and "up" in subdomain and record_type == "A":
                    return "UP"

        return ""

    
    def _check_timediff(self, logline: dict, tracker: dict, threshold: int, count: int):
        ip = extract_query_ip(logline)
        domain = get_registered_domain(logline)
        if not ip or not domain:
            return

        current_ts = extract_timestamp_date(logline)

        # Initialize tracker for this IP if not exists
        if ip not in tracker:
            tracker[ip] = []

        # Add current logline
        tracker[ip].append(logline)

        # Remove old entries (> threshold seconds)
        while tracker[ip] and (current_ts - extract_timestamp_date(tracker[ip][0])).total_seconds() > threshold:
            tracker[ip].pop(0)

        # Alert if enough packets in window
        if len(tracker[ip]) >= count:
            if domain not in self.alarms:
                self.alarms.add(domain)

    def detect(self, logline: dict):

        domain = get_registered_domain(logline)
        if not domain:
            return

        # IODINE SIGNATURE RULES
        if self._is_iodine_packet(logline) == "NULL":
            if not domain in self.alarms:
                self.alarms.add(domain)
        if self._is_iodine_packet(logline) == "CNAME":
            self._check_timediff(logline, self.iodine_cname_tracker, 15, 2)

        # DNSCAT2 SIGNATURE RULES
        if self._is_dnscat2_packet(logline) == "PLAIN":
            self._check_timediff(logline, self.dnscat_plain_tracker, 10, 2)
        if self._is_dnscat2_packet(logline) == "CNAME":
            self._check_timediff(logline, self.dnscat_cname_tracker, 10, 2)
        if self._is_dnscat2_packet(logline) == "MX":
            self._check_timediff(logline, self.dnscat_mx_tracker, 10, 2)
        if self._is_dnscat2_packet(logline) == "TXT":
            self._check_timediff(logline, self.dnscat_txt_tracker, 10, 2)

        # DNS2TCP SIGNATURE RULES
        if self._is_dns2tcp_packet(logline):
            if not domain in self.alarms:
                self.alarms.add(domain)

        # THUNDERDNS SIGNATURE RULES
        if self._is_thunderdns_packet(logline):
            self._check_timediff(logline, self.thunderdns_tracker, 2, 15)

        # OZYMAN DNS SIGNATURE RULES
        if self._is_ozymandns_packet(logline) == "UP":
            self._check_timediff(logline, self.ozyman_up_tracker, 5, 20)
        if self._is_ozymandns_packet(logline) == "DOWN":
            self._check_timediff(logline, self.ozyman_down_tracker, 5, 20)