from detection.detector_base.detector_base import Detector
from detection.detectors.RealTrafficDNS.config import MODEL_DIR,VARIANT,THRESHOLD,MAX_GROUPS,MAX_QUERIES_PER_GROUP
from detection.detectors.RealTrafficDNS.features import group_key,curate
from detection.detectors.RealTrafficDNS.model import RealTrafficModel


class RealTrafficDNS(Detector):
    def __init__(self):
        super().__init__();self.model=RealTrafficModel.load(MODEL_DIR)
        self.groups={};self.invalid=set();self.day=None
        self.first_day=True
        self.group_counts=dict(classified=0,curation_dropped=0,capacity_dropped=0,invalid_events=0,partial_day=0)

    def reset(self):
        super().reset();self.groups.clear();self.invalid.clear();self.day=None
        self.first_day=True
        self.group_counts={key:0 for key in self.group_counts}

    def _finalize(self,partial=False):
        for key,names in self.groups.items():
            if key in self.invalid:continue
            group=curate(names)
            if group is None:self.group_counts['curation_dropped']+=1;continue
            scores=self.model.scores(group)
            self.group_counts['classified']+=1
            self.group_counts['partial_day']+=int(partial)
            # CNN is majority voting: ties are benign; RF threshold is separate.
            malicious=scores['cnn']>.5 if VARIANT=='cnn' else scores[VARIANT]>=THRESHOLD
            if malicious:self.alarms.add(key[0])
        self.groups.clear();self.invalid.clear()

    def detect(self,logline):
        if logline.get('dns',{}).get('flags',{}).get('qr'):return
        try:
            name=logline['dns']['qname']
            key=group_key(name,logline['network']['query-ip'],logline['dnstap']['timestamp-rfc3339ns'])
            if key is None:return
        except (KeyError,ValueError,TypeError,AttributeError):self.group_counts['invalid_events']+=1;return
        if self.day is not None and key[2]<self.day:
            self.group_counts['invalid_events']+=1;return
        if self.day is not None and key[2]>self.day:
            self._finalize(partial=self.first_day)
            self.first_day=False
        self.day=key[2]
        if key in self.invalid:return
        if key not in self.groups and len(self.groups)>=MAX_GROUPS:
            self.group_counts['capacity_dropped']+=1;return
        rows=self.groups.setdefault(key,[])
        if len(rows)>=MAX_QUERIES_PER_GROUP:
            self.invalid.add(key);self.groups[key]=[];self.group_counts['capacity_dropped']+=1;return
        rows.append(name)

    def evaluate(self,*args,**kwargs):
        # The original benchmark calls evaluate at its cutoff, not necessarily
        # UTC midnight. Explicitly record these partial-day decisions.
        self._finalize(partial=True)
        result=super().evaluate(*args,**kwargs)
        result.update(group_statistics=dict(self.group_counts),model_variant=VARIANT,
                      decision_unit='registered_domain_client_UTC_day',
                      reproduction_status='paper_based_reconstruction')
        return result
