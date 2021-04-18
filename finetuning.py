import time

import numpy
import torch
from sklearn.model_selection import train_test_split

import glob
import re
from sklearn.metrics import accuracy_score, classification_report

from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import BertForSequenceClassification, AdamW


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


labels = []
sentences = []
for file in glob.glob('./txt/*.txt', recursive=True):
    header = ""
    print(file)
    with open(file, 'r') as f:
        count = 0
        sents = re.split(r'\.\s', f.read())
        for sentence in sents:
            if "Краткая информация" in sentence:
                header = 0
            elif "Диагностика" in sentence:
                header = 1
            elif "Лечение" in sentence:
                header = 2
            elif "Реабилитация" in sentence:
                header = 3
            elif "Профилактика" in sentence:
                header = 4
            if header != '':
                labels.append(header)
                sentences.append(sentence.replace('\n', ' '))

tokenizer = DistilBertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased-sentence', model_max_length=512)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
y_test = torch.tensor(labels, dtype=torch.long)
val_encodings = tokenizer(sentences, truncation=True, padding=True)

val_dataset = MyDataset(val_encodings, y_test)
model = torch.load("model-final2", map_location=torch.device('cpu'))
model.to(device)
val_loader = DataLoader(val_dataset, batch_size=4)
model.eval()

y = numpy.array([])
y_pred = numpy.array([])
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        print(i)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        y = numpy.append(y, labels.cpu().detach().numpy())
        y_pred = numpy.append(y_pred, numpy.argmax(outputs[0].cpu().detach().numpy(), 1))

print(accuracy_score(y, y_pred))
print(classification_report(y, y_pred))

# labels = []
# sentences = []
# for file in glob.glob('./txt/*.txt', recursive=True):
#     header =""
#     print(file)
#     current = ""
#     with open(file, 'r') as f:
#         count = 0
#         sents = re.split(r'\.\s', f.read())
#         for sentence in sents:
#             if "Краткая информация" in sentence:
#                 header = 0
#             elif "Диагностика" in sentence:
#                 header = 1
#             elif "Лечение" in sentence:
#                 header = 2
#             elif "Реабилитация" in sentence:
#                 header = 3
#             elif "Профилактика" in sentence:
#                 header = 4
#             if header != '':
#                 labels.append(header)
#                 sentences.append(current + " " + sentence.replace('\n', ' '))
#                 current = sentence.replace('\n', ' ')
#
# X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.3)
# y_train = torch.tensor(y_train, dtype=torch.long)
# y_test = torch.tensor(y_test, dtype=torch.long)

# tokenizer = DistilBertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased-sentence', model_max_length=512)
#
# train_encodings = tokenizer(X_train, truncation=True, padding=True)
# val_encodings = tokenizer(X_test, truncation=True, padding=True)
#
# print(train_encodings[0])
#
# train_dataset = MyDataset(train_encodings, y_train)

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model = BertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased-sentence', num_labels=5)
# model.to(device)
# model.train()
#
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
#
# optim = AdamW(model.parameters(), lr=3e-5)
#
# start_time = time.time()
# for epoch in range(3):
#     for i, batch in enumerate(train_loader):
#         optim.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask)
#         loss = F.cross_entropy(outputs.logits, labels)
#         loss.backward()
#         optim.step()
#         print(i, loss.data.item())
#         if i % 1000 == 0:
#             torch.save(model, "ckp-" + str(i))
#
#     running_loss = 0.0
#     start_time = time.time()
# torch.save(model, "model-final2")
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
