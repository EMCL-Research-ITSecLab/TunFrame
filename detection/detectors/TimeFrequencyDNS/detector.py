"""Event-time, full-window adapter. Ports intentionally do not split flows."""
from collections import defaultdict,deque
import math
from urllib.parse import SplitResult
import tldextract

from detection.detector_base.detector_base import Detector
from detection.detectors.TimeFrequencyDNS.config import MODEL_PATH,VARIANT,MAX_PACKETS
from detection.detectors.TimeFrequencyDNS.features import from_collector,window_features
from detection.detectors.TimeFrequencyDNS.model import TimeFrequencyModel

_suffixes=tldextract.TLDExtract(suffix_list_urls=(),cache_dir=None)


class TimeFrequencyDNS(Detector):
    def __init__(self):
        super().__init__();self.model=TimeFrequencyModel.load(MODEL_PATH)
        self.history=defaultdict(deque);self.next_end=None;self.watermark=float('-inf');self.retained=0
        self.discard_until=float('-inf')
        self.counts=dict(windows=0,alerted_windows=0,excluded=0,late_packets=0,capacity_dropped=0,incomplete_windows=0)

    def reset(self):
        super().reset();self.history.clear();self.next_end=None;self.watermark=float('-inf');self.retained=0
        self.discard_until=float('-inf')
        self.counts={key:0 for key in self.counts}

    def advance(self,timestamp):
        if self.next_end is None:return
        if timestamp<self.watermark:raise ValueError('Cannot move the observation clock backwards')
        self.watermark=timestamp
        while self.next_end<=timestamp:
            start=self.next_end-self.model.window
            for flow in list(self.history):
                packets=self.history[flow]
                while packets and packets[0].timestamp<start:packets.popleft();self.retained-=1
                if not packets:del self.history[flow];continue
                if self.next_end<=self.discard_until:
                    self.counts['incomplete_windows']+=1;continue
                features=window_features(packets,start,self.next_end,self.model.fine,self.model.lexicon)
                if features is not None:
                    self.counts['windows']+=1
                    if self.model.score(features,VARIANT)>=0:
                        self.counts['alerted_windows']+=1
                        # Native decision unit is an IP pair/window, not a domain.
                        # Project that alert onto query domains for the legacy API.
                        for packet in packets:
                            if packet.response or packet.timestamp>=self.next_end:continue
                            for name in packet.names:
                                domain=_suffixes.extract_urllib(SplitResult('',name,'','','')).top_domain_under_public_suffix.lower()
                                if domain:self.alarms.add(domain)
            self.next_end+=self.model.step
            if not self.history and self.next_end<=timestamp:
                self.next_end+=(math.floor((timestamp-self.next_end)/self.model.step)+1)*self.model.step

    def detect(self,event):
        try:packet=from_collector(event)
        except (KeyError,ValueError,TypeError,OverflowError):self.counts['excluded']+=1;return
        if packet.timestamp<self.watermark:self.counts['late_packets']+=1;return
        self.watermark=packet.timestamp
        if self.next_end is None:self.next_end=packet.timestamp+self.model.window
        self.advance(packet.timestamp)  # Close [start,end) before inserting the boundary packet.
        if self.retained>=MAX_PACKETS:
            self.discard_until=packet.timestamp+self.model.window
            self.counts['capacity_dropped']+=1;return
        self.history[packet.flow].append(packet);self.retained+=1

    def evaluate(self,*args,**kwargs):
        result=super().evaluate(*args,**kwargs)
        result.update(window_statistics=dict(self.counts,pending_flows=len(self.history)),
            model_variant=VARIANT,window_seconds=self.model.window,
            observation_coverage='full_windows_evaluated' if self.counts['windows'] else 'no_full_windows_evaluated',
            native_decision_unit='client_server_IP_pair_window',
            domain_projection='all_query_domains_in_alerted_windows',
            reproduction_status='paper_based_reconstruction')
        return result
