"""Adapter for explicitly enriched collector events; never guesses features."""
import joblib
import numpy as np
from collections import Counter
import math
from detection.detector_base.detector_base import Detector
from detection.detector_base.feature_extraction import get_registered_domain
from detection.detectors.TabuPIO.model import length_alarm


class TabuPIO(Detector):

    def __init__(self, settings=None):
        super().__init__()
        self.model = None
        self.variant = 'combined'
        self.event_predictions = []
        if settings is None:
            from detection.detectors.TabuPIO.config import MODEL_PATH, LENGTH_LAYER
            settings = dict(model=MODEL_PATH, length_layer=LENGTH_LAYER, variant='combined')
        self.configure(settings)

    def configure(self, settings):
        self.variant = settings.get('variant', 'combined')
        if self.variant not in ('combined', 'length_only', 'classifier_only'):
            raise ValueError('Invalid TabuPIO variant')
        self.length_layer = settings.get('length_layer', 'unresolved')
        if self.variant != 'length_only':
            if not settings.get('model'):
                raise ValueError('TabuPIO requires a trained model path')
            artifact = joblib.load(settings['model'])
            if artifact.get('schema_version') != 1:
                raise ValueError('Unsupported TabuPIO model schema')
            self.model = artifact['model']
            if self.variant == 'combined' and self.model.length_layer != self.length_layer:
                raise ValueError('Configured length layer differs from the trained artifact')
        if self.variant != 'classifier_only' and self.length_layer not in ('dns','ip','frame'):
            raise ValueError('Length-based modes require explicit dns/ip/frame length layer')

    def reset(self):
        super().reset()
        self.event_predictions = []

    def detect(self, logline):
        observation = logline.get('tabu_pio', {})
        if not observation and 'observation' in logline:
            shared = logline['observation']
            observation = dict(features=shared.get('features', {}),
                               length_layer=self.length_layer,
                               packet_length=shared.get('packet_lengths', {}).get(self.length_layer))
        if not observation:
            dns=logline.get('dns', {})
            codes={'A':1,'NS':2,'CNAME':5,'SOA':6,'NULL':10,'PTR':12,'MX':15,'TXT':16,'AAAA':28,'SRV':33,'OPT':41,'ANY':255}
            values={}
            for field in ('qdcount','ancount','arcount','nscount'):
                if field in dns:values[field]=dns[field]
            if dns.get('qtype') in codes:values['qd_qtype']=codes[dns['qtype']]
            if isinstance(dns.get('qname'),str):
                name=(dns['qname'].rstrip('.')+'.').encode('utf8')
                values['qd_qname_len']=len(name)
                values['qd_qname_shannon']=-sum((n/len(name))*math.log2(n/len(name)) for n in Counter(name).values())
            records=dns.get('resource-records',{})
            for prefix,count,section in [('an','ancount','an'),('ar','arcount','ar')]:
                if dns.get(count)==0:
                    for suffix in ['rrname_len','rrname_shannon','rrname_shanonn','type','ttl','rdata_len','rdata_shannon']:
                        values[f'{prefix}_{suffix}']=0
                elif records.get(section):
                    record=records[section][0]
                    if record.get('rdatatype',record.get('type')) in codes:
                        values[f'{prefix}_type']=codes[record.get('rdatatype',record.get('type'))]
                    if 'ttl' in record:values[f'{prefix}_ttl']=record['ttl']
            observation=dict(features=values,length_layer='dns',packet_length=dns.get('length'))
        length = None
        if self.variant != 'classifier_only':
            if observation.get('length_layer') != self.length_layer:
                raise ValueError('Missing or mismatched tabu_pio.length_layer')
            length = observation.get('packet_length')
            if length is None:
                raise ValueError('Missing tabu_pio.packet_length; DNS name length is not a substitute')
        if self.variant == 'length_only':
            predicted = int(length_alarm([length])[0])
        elif self.variant == 'combined' and length_alarm([length])[0]:
            predicted = 1
        else:
            values = observation.get('features', {})
            try:
                x = np.array([[values.get(name, float('nan')) for name in self.model.feature_names]], dtype=float)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Event lacks the trained feature schema; enrich it before detection') from exc
            predicted = int(self.model.predict(x, variant='classifier_only')[0])
        self.event_predictions.append(dict(timestamp=logline.get('dnstap', {}).get('timestamp-rfc3339ns'),
                                           prediction=predicted))
        if predicted:
            domain = get_registered_domain(logline)
            if domain:
                self.alarms.add(domain)
