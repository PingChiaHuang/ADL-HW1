# Sample Code for Homework 1 ADL NTU 109 Spring

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Download my model
```shell
bash download.sh
```

## Intent detection
```shell
python train_intent.py
```

## Intent detection (my model)
```shell
python train_intent.py --apply_init
```

## Slot detection
```shell
python train_slot.py
```

## Slot detection (my model)
```shell
python train_slot.py --apply_init --add_labels
```