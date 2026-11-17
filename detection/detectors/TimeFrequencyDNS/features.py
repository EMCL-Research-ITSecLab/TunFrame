"""ICC 2023 Table II statistics and Equations 2–3 FFT magnitudes."""
from collections import Counter,defaultdict,deque
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import math
import re
import socket

import dpkt
import numpy as np

TIME_FEATURES=('mean_name_length','mean_special_count','mean_special_fraction','mean_name_entropy',
               'mean_word_count','mean_longest_word','mean_label_count','mean_dns_length',
               'packet_count','upload_download_byte_ratio','request_rate','query_response_count_ratio','mean_rtt')


class Lexicon:
    def __init__(self,words):
        self.words=sorted(set(w.strip().lower() for w in words if w.strip().isascii() and w.strip().isalpha() and len(w.strip())>=2))
        if not self.words:raise ValueError('Need an explicit English word dictionary')
        self.trie={}
        for word in self.words:
            node=self.trie
            for c in word:node=node.setdefault(c,{})
            node['']=True

    @lru_cache(maxsize=16384)
    def name_features(self,name):
        name=name.rstrip('.')
        n=len(name)
        counts=Counter(name)
        entropy=-sum((v/n)*math.log2(v/n) for v in counts.values()) if n else 0.
        special=sum(not(c.isascii() and c.isalnum()) and c!='.' for c in name)
        count=longest=0
        # Greedy longest dictionary words inside alphabetic runs. The paper does
        # not release its dictionary or word-segmentation procedure.
        for run in re.findall('[a-z]+',name.lower()):
            i=0
            while i<len(run):
                node=self.trie;j=i;end=i
                while j<len(run) and run[j] in node:
                    node=node[run[j]];j+=1
                    if '' in node:end=j
                if end>i:count+=1;longest=max(longest,end-i);i=end
                else:i+=1
        return (n,special,special/n if n else 0.,entropy,count,longest,name.count('.')+1 if n else 0)


@dataclass(frozen=True)
class Packet:
    timestamp: float
    client: str
    server: str
    response: bool
    length: int
    names: tuple
    transaction: tuple

    @property
    def flow(self):return self.client,self.server


def from_collector(event):
    dns,network=event['dns'],event['network']
    if dns.get('malformed-packet'):raise ValueError('Malformed DNS')
    name=dns['qname']
    if not isinstance(name,str) or not name:raise ValueError('Missing name')
    if dns['qdcount']!=1:raise ValueError('Collector exposes one question only')
    stamp=datetime.fromisoformat(event['dnstap']['timestamp-rfc3339ns'].replace('Z','+00:00')).timestamp()
    length=int(dns['length'])
    if length<12:raise ValueError('Truncated DNS header')
    transaction=(network['protocol'].upper(),int(network['query-port']),int(network['response-port']),
                 int(dns['id']),((name.rstrip('.').lower(),str(dns['qtype'])),))
    return Packet(stamp,str(network['query-ip']),str(network['response-ip']),bool(dns['flags']['qr']),length,(name.rstrip('.'),),transaction)


def from_wire(stamp,data,linktype):
    if linktype==1:network=dpkt.ethernet.Ethernet(data).data
    elif linktype==113:network=dpkt.sll.SLL(data).data
    elif linktype in (101,228,229):network=dpkt.ip.IP(data) if data[0]>>4==4 else dpkt.ip6.IP6(data)
    else:raise ValueError(f'Unsupported link type {linktype}')
    if not isinstance(network,(dpkt.ip.IP,dpkt.ip6.IP6)):return None
    if isinstance(network,dpkt.ip.IP) and (network.mf or network.offset):
        raise ValueError('Fragment reassembly required')
    transport=network.data
    if not isinstance(transport,(dpkt.udp.UDP,dpkt.tcp.TCP)):return None
    if transport.sport!=53 and transport.dport!=53:return None
    payload=transport.data
    protocol='TCP' if isinstance(transport,dpkt.tcp.TCP) else 'UDP'
    if protocol=='TCP':
        if len(payload)<2 or int.from_bytes(payload[:2],'big')+2!=len(payload):
            raise ValueError('Complete single TCP DNS frame required')
        payload=payload[2:]
    dns=dpkt.dns.DNS(payload)
    family=socket.AF_INET if isinstance(network,dpkt.ip.IP) else socket.AF_INET6
    source,dest=socket.inet_ntop(family,network.src),socket.inet_ntop(family,network.dst)
    response=bool(dns.qr)
    client,server=(dest,source) if response else (source,dest)
    cp,sp=(transport.dport,transport.sport) if response else (transport.sport,transport.dport)
    names=tuple(q.name.rstrip('.') for q in dns.qd)
    # Numeric qtype is not needed in model features; stringify names consistently
    # within wire input. Collector uses mnemonic types in its own matching path.
    transaction=(protocol,cp,sp,dns.id,tuple((q.name.rstrip('.').lower(),str(q.type)) for q in dns.qd))
    return Packet(float(stamp),client,server,response,len(payload),names,transaction)


def window_features(packets,start,end,fine,lexicon):
    duration=end-start
    bins=int(round(duration/fine))
    if bins<2 or not math.isclose(bins*fine,duration,abs_tol=1e-6):raise ValueError('Coarse window must divide into fine bins')
    values=[p for p in packets if start<=p.timestamp<end]
    if not values:return None
    queries=[p for p in values if not p.response];responses=[p for p in values if p.response]
    names=[name for p in queries for name in p.names]
    lexical=np.mean([lexicon.name_features(name) for name in names],axis=0) if names else np.zeros(7)
    up=sum(p.length for p in queries);down=sum(p.length for p in responses)
    pending=defaultdict(deque);delays=[]
    for p in values:
        if not p.response:pending[p.transaction].append(p.timestamp)
        elif pending[p.transaction]:delays.append(p.timestamp-pending[p.transaction].popleft())
    time=np.r_[lexical,np.mean([p.length for p in values]),len(values),up/down if down else 0.,
               len(queries)/duration,len(queries)/len(responses) if responses else 0.,np.mean(delays) if delays else 0.]
    totals=np.zeros(bins);counts=np.zeros(bins)
    for p in values:
        index=min(bins-1,int((p.timestamp-start)/fine));totals[index]+=p.length;counts[index]+=1
    sequence=np.divide(totals,counts,out=np.zeros(bins),where=counts!=0)
    frequency=np.abs(np.fft.fft(sequence))
    result=np.r_[time,frequency]
    if not np.isfinite(result).all():raise ValueError('Non-finite features')
    return result
