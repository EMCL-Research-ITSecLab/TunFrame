"""Optional improved variant. Original author weights are read, never overwritten."""
import argparse
import csv
import hashlib
import io
from pathlib import Path
import random
import zipfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import get_scheduler

from detection.detectors.TBDMfDCC.config import MODEL_DIR, IMPROVED_MODEL_DIR, IMPROVED_MAX_LENGTH
from detection.detectors.TBDMfDCC.model import DNSClassifier


def train(archive, output, dataset, seed, threads):
    if output.resolve() == MODEL_DIR.resolve():
        raise ValueError('Improved training must not overwrite the author model')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty improved-model output directory')
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with zipfile.ZipFile(archive) as bundle:
        contents = bundle.read('huggingface_test/'+dataset)
    rows = list(csv.reader(io.StringIO(contents.decode('ISO-8859-1'))))
    if not rows or any(len(r)!=2 or r[1].strip() not in ('0','1') for r in rows):
        raise ValueError('Expected domain,label CSV without a header')
    data = pd.DataFrame({'content':[r[0] for r in rows], 'label':[int(r[1]) for r in rows]})
    training, rest = train_test_split(data, test_size=.4, random_state=10, stratify=data.label)
    validation, test = train_test_split(rest, test_size=.5, random_state=10, stratify=rest.label)
    classifier = DNSClassifier(MODEL_DIR)
    tokenizer, model = classifier.tokenizer, classifier.model
    initial_hash = hashlib.sha256((MODEL_DIR/'pytorch_model.bin').read_bytes()).hexdigest()
    def loader(frame, shuffle=False):
        encoded = tokenizer(frame.content.tolist(), padding='max_length', truncation=True,
                            max_length=IMPROVED_MAX_LENGTH, return_tensors='pt')
        tensors = TensorDataset(encoded['input_ids'],encoded['attention_mask'],torch.tensor(frame.label.tolist()))
        return DataLoader(tensors,batch_size=16,sampler=RandomSampler(tensors) if shuffle else None)
    batches = loader(training,True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5)
    scheduler = get_scheduler('linear',optimizer,num_warmup_steps=0,num_training_steps=2*len(batches))
    # The held-out test set does not participate in fitting or epoch selection.
    for epoch in range(2):
        model.train()
        losses = []
        for step,(ids,mask,labels) in enumerate(batches):
            optimizer.zero_grad()
            loss = model(input_ids=ids,attention_mask=mask,labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();scheduler.step()
            losses.append(loss.item())
            if (step+1)%25==0:print(f'Epoch {epoch+1}, batch {step+1}/{len(batches)}',flush=True)
        model.eval()
        correct = count = 0
        with torch.inference_mode():
            for ids,mask,labels in loader(validation):
                correct += int((model(input_ids=ids,attention_mask=mask).logits.argmax(-1)==labels).sum())
                count += len(labels)
        print(f'Epoch {epoch+1}: loss={np.mean(losses):.6f}, validation accuracy={correct/count:.4f}',flush=True)
    model.config.id2label={0:'Normal',1:'Covert'}
    model.config.label2id={'Normal':0,'Covert':1}
    output.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(output,safe_serialization=True)
    tokenizer.save_pretrained(output)
    # Compare both model variants on identical source rows, using each variant's
    # declared inference length. Author checkpoint remains at 256 tokens.
    original = DNSClassifier(MODEL_DIR,256)
    improved = DNSClassifier(output,IMPROVED_MAX_LENGTH)
    report = test.copy()
    report.insert(0,'source_row',report.index)
    for name,predictor in [('authors',original),('improved',improved)]:
        prediction = []
        for start in range(0,len(test),16):
            prediction.extend(predictor.predict(test.content.tolist()[start:start+16]))
        report[name]=prediction
    # Verify saved weights reproduce the trained model, not just its serializer.
    memory_predictions=[]
    model.eval()
    with torch.inference_mode():
        for ids,mask,_ in loader(test):
            memory_predictions.extend(model(input_ids=ids,attention_mask=mask).logits.argmax(-1).tolist())
    assert report.improved.tolist()==memory_predictions
    assert hashlib.sha256((MODEL_DIR/'pytorch_model.bin').read_bytes()).hexdigest()==initial_hash
    results = Path(__file__).resolve().parent/'results'
    results.mkdir(exist_ok=True)
    report.to_csv(results/'comparison.csv',index=False)
    summary=[f'Dataset: {dataset}; SHA-256: {hashlib.sha256(contents).hexdigest()}',
             f'Train/validation/test: {len(training)}/{len(validation)}/{len(test)}; split seed=10; training seed={seed}',
             f'Exact DNS-name overlap, training/test: {len(set(training.content) & set(test.content))}; source split retained.',
             'Improved: two epochs, AdamW 2e-5, batch=16, length=64 in all phases, schedule=2 epochs.',
             'This is an improved variant, not an exact paper reproduction.',
             f'Original checkpoint unchanged: {initial_hash}']
    for name in ['authors','improved']:
        tn,fp,fn,tp = confusion_matrix(report.label,report[name],labels=[0,1]).ravel()
        summary.append(f'{name}: TP={tp}, FP={fp}, TN={tn}, FN={fn}; accuracy={(tp+tn)/len(test):.6f}')
    (results/'comparison.txt').write_text('\n'.join(summary)+'\n')
    print('\n'.join(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=Path('TBDMfDCC.zip'))
    parser.add_argument('--output',type=Path,default=IMPROVED_MODEL_DIR)
    parser.add_argument('--dataset',choices=['Dataset_small.csv','Dataset.csv'],default='Dataset_small.csv')
    parser.add_argument('--seed',type=int,default=10)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    train(args.archive,args.output,args.dataset,args.seed,args.threads)
