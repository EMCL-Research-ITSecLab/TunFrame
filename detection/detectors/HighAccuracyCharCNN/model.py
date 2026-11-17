"""The final Char-CNN from Chen & Li, CIS 2020, Table II."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

LENGTH = 1014
# The paper omits its alphabet. Printable ASCII preserves payload case; reserve
# 0 for end/padding and 1 for unknown characters. Channel widths follow the
# small Char-CNN of the cited Zhang et al. architecture, not a reported setting.
ALPHABET = ''.join(chr(i) for i in range(32, 127))


def encode(names):
    lookup = {c: i+2 for i,c in enumerate(ALPHABET)}
    rows = []
    for name in names:
        ids = [lookup.get(c,1) for c in name.rstrip('.')[:LENGTH]]
        rows.append(ids+[0]*(LENGTH-len(ids)))
    return torch.tensor(rows,dtype=torch.long)


class CharCNN(nn.Module):
    def __init__(self, channels=256, hidden=1024):
        super().__init__()
        self.channels, self.hidden = channels, hidden
        blocks = []
        incoming = len(ALPHABET)+2
        length = LENGTH
        for index,kernel in enumerate([7,7,3,3,3,3]):
            blocks += [nn.Conv1d(incoming,channels,kernel),nn.ReLU()]
            length -= kernel-1
            if index in (0,1,5):
                blocks += [nn.MaxPool1d(3)]
                length //= 3
            incoming = channels
        self.convolutions = nn.Sequential(*blocks)
        self.dense = nn.Sequential(nn.Linear(channels*length,hidden),nn.BatchNorm1d(hidden),nn.ReLU(),nn.Dropout(.5),
            nn.Linear(hidden,hidden),nn.BatchNorm1d(hidden),nn.ReLU(),nn.Dropout(.5),
            nn.Linear(hidden,1),nn.BatchNorm1d(1))

    def forward(self, tokens):
        # End tokens have a zero character vector; unknowns retain their own channel.
        value = F.one_hot(tokens,num_classes=len(ALPHABET)+2).float()
        value = value*(tokens!=0).unsqueeze(-1)
        return self.dense(self.convolutions(value.transpose(1,2)).flatten(1)).squeeze(-1)

    def probabilities(self, names, batch=32):
        self.eval()
        values = []
        with torch.inference_mode():
            for start in range(0,len(names),batch):
                values.extend(self(encode(names[start:start+batch])).sigmoid().cpu().tolist())
        return values

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(schema_version=1,channels=self.channels,hidden=self.hidden,
                        alphabet=ALPHABET,length=LENGTH,state=self.state_dict()),path)

    @classmethod
    def load(cls,path):
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['alphabet']!=ALPHABET or state['length']!=LENGTH:
            raise ValueError('Character encoding differs from trained model')
        model=cls(state['channels'],state['hidden'])
        model.load_state_dict(state['state'],strict=True)
        return model.eval()
