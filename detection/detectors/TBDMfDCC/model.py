"""Offline DistilBERT inference following the authors' DNS classification code."""
from pathlib import Path
import argparse
import hashlib
import zipfile

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


class DNSClassifier:
    def __init__(self, directory, max_length=256):
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f'Missing TBDMfDCC model: {directory}. Extract the author model first.')
        self.tokenizer = DistilBertTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = DistilBertForSequenceClassification.from_pretrained(directory, local_files_only=True)
        if self.model.config.num_labels != 2:
            raise ValueError('TBDMfDCC requires two labels: Normal=0, Covert=1')
        self.model.to('cpu').eval()
        self.max_length = max_length

    def predict(self, domains):
        if not domains:
            return []
        # Preserve complete qnames; WordPiece performs its own uncased normalization.
        encoded = self.tokenizer(domains, padding='max_length', truncation=True,
                                 max_length=self.max_length, return_tensors='pt')
        with torch.inference_mode():
            logits = self.model(input_ids=encoded['input_ids'],
                                attention_mask=encoded['attention_mask']).logits
        return logits.argmax(dim=-1).cpu().tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract the unchanged author model from TBDMfDCC.zip')
    parser.add_argument('--archive', type=Path, default=Path('TBDMfDCC.zip'))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'model')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty model output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as bundle:
        for name in ('config.json', 'vocab.txt', 'pytorch_model.bin'):
            data = bundle.read('huggingface_test/'+name)
            (args.output/name).write_bytes(data)
            print(name, hashlib.sha256(data).hexdigest())
