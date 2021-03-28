import pandas as pd
import sys
import json


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

column = "intent" if "intent" in dict_1 else "tags"
if column == "intent":
    result = sum([label_1 == label_2 for label_1, label_2 in zip(dict_1[column], dict_2[column])])
else:
    result = sum(
        [" ".join(label_1) == label_2 for label_1, label_2 in zip(dict_1[column], dict_2[column])]
    )
print(f"{result} / {len(dict_1[column])} = {result / len(dict_1[column])}")
