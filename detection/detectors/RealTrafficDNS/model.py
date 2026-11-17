"""Paper multichannel prefix CNN, RF and RF+CNN ensemble."""
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn

LENGTH=253  # Not specified in the paper; bounded by ordinary DNS name length.
ALPHABET=''.join(chr(i) for i in range(32,127))


def encode(prefixes):
    lookup={c:i+2 for i,c in enumerate(ALPHABET)}
    result=[]
    for prefix in prefixes:
        row=[lookup.get(c,1) for c in prefix[:LENGTH]]
        result.append(row+[0]*(LENGTH-len(row)))
    return torch.tensor(result,dtype=torch.long)


class PrefixCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(len(ALPHABET)+2,32,padding_idx=0)
        self.convolutions=nn.ModuleList([nn.Conv1d(32,256,kernel) for kernel in [2,3,4,5]])
        self.hidden=nn.Sequential(nn.Linear(1024,1024),nn.ReLU(),nn.Linear(1024,1024),nn.ReLU(),nn.Linear(1024,1))

    def forward(self,tokens):
        values=self.embedding(tokens).transpose(1,2)
        pooled=torch.cat([layer(values).relu().amax(-1) for layer in self.convolutions],dim=1)
        return self.hidden(pooled).squeeze(-1)

    def probabilities(self,prefixes):
        self.eval();values=[]
        with torch.inference_mode():
            for i in range(0,len(prefixes),128):values.extend(self(encode(prefixes[i:i+128])).sigmoid().tolist())
        return values


class RealTrafficModel:
    def __init__(self,cnn,rf,combined):self.cnn,self.rf,self.combined=cnn.eval(),rf,combined

    def scores(self,group):
        from detection.detectors.RealTrafficDNS.features import statistics
        probabilities=self.cnn.probabilities(group['prefixes'])
        features=np.asarray([statistics(group,probabilities)])
        return dict(cnn=float(np.mean(np.asarray(probabilities)>=.5)),
                    rf=float(self.rf.predict_proba(np.delete(features,7,axis=1))[0,1]),
                    rf_cnn=float(self.combined.predict_proba(features)[0,1]))

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        torch.save(dict(alphabet=ALPHABET,length=LENGTH,state=self.cnn.state_dict()),directory/'cnn.pt')
        joblib.dump(self.rf,directory/'rf.joblib');joblib.dump(self.combined,directory/'rf_cnn.joblib')

    @classmethod
    def load(cls,directory):
        directory=Path(directory);state=torch.load(directory/'cnn.pt',map_location='cpu',weights_only=True)
        if state['alphabet']!=ALPHABET or state['length']!=LENGTH:raise ValueError('Incompatible prefix encoding')
        cnn=PrefixCNN();cnn.load_state_dict(state['state'],strict=True)
        return cls(cnn,joblib.load(directory/'rf.joblib'),joblib.load(directory/'rf_cnn.joblib'))
