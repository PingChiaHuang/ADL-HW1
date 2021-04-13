# ADL 109 Spring HW1

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