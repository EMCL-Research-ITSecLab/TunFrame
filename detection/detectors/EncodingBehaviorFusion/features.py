"""Paper Algorithm 1, Table 3 and bounded DNS query/response pairing."""
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
import math
from urllib.parse import SplitResult

import tldextract

_suffixes = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
STATISTICS = ('query_payload_length', 'response_payload_length', 'mean_answer_ttl',
              'answers_per_question', 'response_delay_seconds', 'upload_download_ratio')
# 38 specified characters + shared padding/unknown ID = 39 entries, indices 0..38.
ALPHABET = '0123456789abcdefghijklmnopqrstuvwxyz_\t'


def registered_domain(name):
    return _suffixes.extract_urllib(SplitResult('', name.rstrip('.'), '', '', '')).top_domain_under_public_suffix.lower()


def content(questions, answers):
    queries = []
    for question in questions:
        # Treat a terminal root dot as notation, not an additional DNS label.
        labels = question.rstrip('.').split('.')
        queries.append('.'.join(labels[:-2]) if len(labels) >= 3 else '.'.join(labels))
    text = ''.join(queries) + '\t' + ''.join(a.replace(':', '') for a in answers)
    return text.replace('.', '').lower()


def encode(text, limit):
    lookup = {char: i+1 for i, char in enumerate(ALPHABET)}
    values = [lookup.get(char, 0) for char in text[:limit]]
    return values + [0]*(limit-len(values)), len(values)


@dataclass(frozen=True)
class Observation:
    timestamp: float
    client: tuple
    server: tuple
    protocol: str
    transaction_id: int
    response: bool
    questions: tuple
    answers: tuple
    ttls: tuple
    qdcount: int
    ancount: int
    dns_length: int

    @property
    def key(self):
        # Questions guard against concurrent ID reuse and cross-name collisions.
        return (self.client, self.server, self.protocol, self.transaction_id,
                tuple((name.rstrip('.').lower(), str(kind)) for name, kind in self.questions))


def from_collector(event):
    dns, network = event['dns'], event['network']
    if dns.get('malformed-packet'):
        raise ValueError('Malformed DNS packet')
    # Collector uses client/server role fields in both query and response events.
    records = dns.get('resource-records', {}).get('an', [])
    answers = []
    ttls = []
    for rr in records:
        data = rr.get('rdata', rr.get('data'))
        if data is None or not isinstance(data, (str, list)):
            raise ValueError('Unavailable answer content')
        answers.extend(data if isinstance(data, list) else [data])
        ttls.append(float(rr['ttl']))
    if int(dns['ancount']) != len(records):
        raise ValueError('Incomplete collector answer records')
    if int(dns['qdcount']) != 1:
        raise ValueError('Collector schema exposes only one question')
    timestamp = datetime.fromisoformat(event['dnstap']['timestamp-rfc3339ns'].replace('Z', '+00:00')).timestamp()
    return Observation(timestamp, (network['query-ip'], int(network['query-port'])),
        (network['response-ip'], int(network['response-port'])), network['protocol'].upper(),
        int(dns['id']), bool(dns['flags']['qr']), ((dns['qname'], str(dns['qtype'])),),
        tuple(answers), tuple(ttls), int(dns['qdcount']), int(dns['ancount']), int(dns['length']))


def from_packet(packet):
    from scapy.layers.dns import DNS, DNSQR, DNSRR, dnsqtypes
    from scapy.layers.inet import IP, UDP, TCP
    from scapy.layers.inet6 import IPv6
    if DNS not in packet or not (IP in packet or IPv6 in packet):
        return None
    layer = packet[IP] if IP in packet else packet[IPv6]
    if UDP in packet:
        transport, protocol = packet[UDP], 'UDP'
    elif TCP in packet:
        transport, protocol = packet[TCP], 'TCP'
        # Scapy does not reassemble streams in PcapReader. Do not count partial DNS.
        framed = bytes(transport.payload)
        if len(framed) < 2 or int.from_bytes(framed[:2], 'big') + 2 != len(framed):
            raise ValueError('TCP DNS requires one complete framed message')
    else:
        return None
    dns = packet[DNS]
    def text(value):
        return value.decode('latin1') if isinstance(value, bytes) else str(value)
    questions = tuple((text(q.qname), str(dnsqtypes.get(q.qtype, q.qtype))) for q in dns.qd if isinstance(q, DNSQR))
    if len(questions) != int(dns.qdcount):
        raise ValueError('Unparsed questions')
    answers, ttls = [], []
    for rr in dns.an:
        if not hasattr(rr, 'type'):
            raise ValueError('Unsupported answer record')
        data = getattr(rr, 'rdata', None)
        if data is None and int(rr.type) == 15:
            data = rr.exchange  # MX exchange, not its integer preference.
        if data is None and int(rr.type) == 33:
            data = rr.target
        if data is None and int(rr.type) == 6:
            data = [rr.mname, rr.rname, str(rr.serial), str(rr.refresh), str(rr.retry), str(rr.expire), str(rr.minimum)]
        if data is None:
            raise ValueError(f'Unsupported answer type {rr.type}')
        answers.extend(text(x) for x in data) if isinstance(data, list) else answers.append(text(data))
        ttls.append(float(rr.ttl))
    if len(ttls) != int(dns.ancount):
        raise ValueError('Unparsed answers')
    source, dest = (layer.src, int(transport.sport)), (layer.dst, int(transport.dport))
    client, server = (dest, source) if dns.qr else (source, dest)
    return Observation(float(packet.time), client, server, protocol, int(dns.id), bool(dns.qr),
                       questions, tuple(answers), tuple(ttls), int(dns.qdcount), int(dns.ancount),
                       len(bytes(dns)) - (2 if protocol == 'TCP' else 0))


def pair_features(query, response):
    delay = response.timestamp - query.timestamp
    if query.qdcount <= 0 or min(query.dns_length, response.dns_length) < 12 or delay < 0:
        raise ValueError('Invalid pair counts, lengths or timestamp order')
    statistics = [query.dns_length-12, response.dns_length-12,
                  sum(response.ttls)/len(response.ttls) if response.ttls else 0.,
                  response.ancount/query.qdcount, delay, query.dns_length/response.dns_length]
    if not all(math.isfinite(v) for v in statistics):
        raise ValueError('Non-finite pair features')
    return dict(content=content([q[0] for q in query.questions], response.answers),
                statistics=statistics, qname=query.questions[0][0])


class Pairer:
    def __init__(self, timeout=30., capacity=100000):
        self.timeout, self.capacity = timeout, capacity
        self.pending = OrderedDict()
        self.watermark = float('-inf')
        self.counts = dict(pairs=0, expired=0, capacity_evicted=0, unmatched_response=0, retransmitted_query=0)

    def add(self, observation):
        self.watermark = max(self.watermark, observation.timestamp)
        # Input captures are ordered; retain first query timestamp on retransmit.
        while self.pending:
            key, query = next(iter(self.pending.items()))
            if self.watermark-query.timestamp <= self.timeout:
                break
            self.pending.pop(key)
            self.counts['expired'] += 1
        if observation.response:
            query = self.pending.pop(observation.key, None)
            if query is None:
                self.counts['unmatched_response'] += 1
                return None
            if observation.timestamp-query.timestamp > self.timeout:
                self.counts['expired'] += 1
                return None
            result = pair_features(query, observation)
            self.counts['pairs'] += 1
            return result
        if observation.key in self.pending:
            self.counts['retransmitted_query'] += 1
        else:
            self.pending[observation.key] = observation
        if len(self.pending) > self.capacity:
            self.pending.popitem(last=False)
            self.counts['capacity_evicted'] += 1
        return None
