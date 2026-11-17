import csv
import torch
import numpy as np
import pandas as pd
# from sklearn.utils import shuffle
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.model_selection import train_test_split        # 用于训练集、验证集和测试集划分
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

label_dic = {
    'Normal': 0,
    'Covert': 1
}


def get_data(file):
    content = []
    label = []
    with open(file, "r", encoding="ISO-8859-1") as f:
        reader = csv.reader(f)
        for line in reader:
            c = line[0]
            l = line[1]
            content.append(c)
            label.append(int(l.strip()))
    return content, label


content, label = get_data('./Dataset_small.csv')
data = pd.DataFrame({"content": content, "label": label})
# data = shuffle(data)
train_dataset, rest_dataset = train_test_split(data,
                                               test_size=0.4,
                                               random_state=10,
                                               stratify=data.label)
validation_dataset, test_dataset = train_test_split(rest_dataset,
                                                    test_size=0.5,
                                                    random_state=10,
                                                    stratify=rest_dataset.label)

# tokenizer 分词器，本质就是词典，对字进行编码
tokenizer = AutoTokenizer.from_pretrained("./")

# 定义分类模型
model = AutoModelForSequenceClassification.from_pretrained("./",
                                                           num_labels=2,        # num_labels类别数量
                                                           ignore_mismatched_sizes=True)

train_encoding = tokenizer(train_dataset.content.to_list(),
                           truncation=True,
                           padding="max_length",
                           max_length=64,
                           return_tensors="pt")
train_label = train_dataset.label.to_list()

# 单个数据读取到批量batch_size读取
batch_size = 128
train = TensorDataset(train_encoding["input_ids"],
                      train_encoding["attention_mask"],
                      torch.tensor(train_label))
train_sampler = RandomSampler(train)
train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=batch_size)

# 验证集
validation_encoding = tokenizer(validation_dataset.content.to_list(),
                                padding="max_length",
                                max_length=256,
                                return_tensors="pt")
validation_label = validation_dataset.label.to_list()

validation = TensorDataset(validation_encoding["input_ids"],
                           validation_encoding["attention_mask"],
                           torch.tensor(validation_label))
validation_sampler = RandomSampler(validation)
validation_dataloader = DataLoader(validation, sampler=validation_sampler, batch_size=batch_size)

# 测试集
test_encoding = tokenizer(test_dataset.content.to_list(),
                          padding="max_length",
                          max_length=256,
                          return_tensors="pt")
test_label = test_dataset.label.to_list()

test = TensorDataset(test_encoding["input_ids"],
                     test_encoding["attention_mask"])
test_dataloader = DataLoader(test, batch_size=batch_size)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# 定义学习率和训练轮数
num_training_steps = 1 * len(train_dataloader)
lr_scheduler = get_scheduler(name="linear",
                             optimizer=optimizer,
                             num_warmup_steps=0,
                             num_training_steps=num_training_steps)

# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# 精度计算
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# 训练函数
def train(epoch):
    model.train()
    total_train_loss = 0
    item_num = 0
    # total_iter = len(train_dataloader)
    epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch}", ncols=100, leave=True, position=0)
    for batch in epoch_iterator:
        # 正向传播
        optimizer.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()
        # 这里添加进度条的描述信息
        epoch_iterator.set_description(
            f"epoch:{epoch} " +
            f"loss: {loss.item():.4f} " +
            f"lr: {lr_scheduler.get_last_lr()[0]:.1e}")

        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # 梯度裁剪

        # 参数更新
        optimizer.step()
        lr_scheduler.step()

        item_num += 1
        # if (iter_num % 100 == 0):
        #     print("epoch: %d, iter_num: %d, loss: %.4f, %.2f%%" %
        #           (epoch, item_num, loss.item(), item_num / total_iter * 100))

    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_dataloader)))


# 验证函数
def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in validation_dataloader:
        with torch.no_grad():
            # 正常传播
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("Accuracy: %.f" % avg_val_accuracy)
    print("Average validating loss: %.4f" % (total_eval_loss / len(validation_dataloader)))


# 测试函数
def trial():
    model.eval()
    testPreds = []
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)

        preds = np.argmax(outputs["logits"], axis=1).numpy().squeeze().tolist()
        testPreds += preds

    return testPreds


if __name__ == '__main__':
    num_epochs = 2
    for epoch in range(num_epochs):
        print()
        print("----------Epoch: %d ----------" % epoch)
        train(epoch)
        validation()

    testPreds = trial()
    test_dataset = test_dataset.reindex(["content", "label", "result"], axis=1).assign(result=testPreds)
    test_dataset.to_csv("testResults.csv", mode='w+', index=False)

'''
for epoch in range(num_epochs):
    # 训练步骤
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 10 == 0 and not step == 0:
            print("step: ", step, "    loss: ", total_loss/(step*batch_size))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()
        outputs = model(b_input_ids,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)
    print("avg_loss: ", avg_train_loss)

    # 验证步骤
    model.eval()
    total_accuracy = 0
    for batch in validation_dataloader:
        with torch.no_grad():
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)
            outputs = model(input_ids,
                            attention_mask=input_mask,
                            labels=labels)
            predicted = np.argmax(outputs["logits"], axis=1).numpy().squeeze()
            accuracy = (predicted == labels).sum()
            total_accuracy += accuracy

    print("Validation accuracy: {}".format(total_accuracy/len(validation_dataloader)))

# 测试步骤
test_data = tokenizer(test_dataset.content.to_list(),
                      padding="max_length",
                      max_length=256,
                      return_tensors="pt").to(device)
test = TensorDataset(test_data["input_ids"], test_data["attention_mask"])
test_dataloader = DataLoader(test, batch_size=batch_size)

model.eval()
testResults = []
with torch.no_grad():
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        predicted = np.argmax(outputs["logits"], axis=1).numpy().squeeze().tolist()
        testResults += predicted

test_dataset = test_dataset.reindex(["result", "label", "content"], axis=1).assign(result=testResults)
test_dataset.to_csv("results.csv", mode='w+', index=False)
'''
