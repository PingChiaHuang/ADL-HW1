import sys
import json

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


def to_dict(path):
    if "json" in path:
        with open(path, "r") as f:
            data = json.load(f)
        return {key: [d[key] for d in data] for key in data[0]}
    else:
        with open(path, "r") as f:
            columns = f.readline().strip().split(",")
            data = [line.strip().split(",") for line in f.readlines()]
            return {key: [d[i] for d in data] for i, key in enumerate(columns)}


dict_1 = to_dict(sys.argv[1])
dict_2 = to_dict(sys.argv[2])

y_true = dict_1["tags"]
y_pred = [tags.split() for tags in dict_2["tags"]]

print(classification_report(y_true, y_pred, scheme=IOB2, mode="strict"))

seq_correct = 0
seq_total = 0
correct = 0
total = 0
for trues, preds in zip(y_true, y_pred):
    if trues == preds:
        seq_correct += 1
        correct += len(trues)
        total += len(trues)
    else:
        for true, pred in zip(trues, preds):
            if true == pred:
                correct += 1
            total += 1
    seq_total += 1

print(f'token accuracy = {correct} / {total} = {correct / total}')
print(f'joint accuracy = {seq_correct} / {seq_total} = {seq_correct / seq_total}')