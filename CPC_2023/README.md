# Confusing Pair Correction

## Requirements

* Python 3.9

* Pytorch 1.8.0
* torchvision 0.9.0

## Dataset

The dataset structure is like

```t
office-home
|_ logs
|  |_ AC.txt
|  |_ ...
|  |_ RP.txt
|_ Art
|  |_ Alarm_clock
|     |_ 0001.png
|     |_ ...
|  ...
|  |_ Webcam
|     |_ 0001.png
|     |_ ...
|_ ...
|_ Clipart
|  |_ Alarm_clock
|     |_ 0001.png
|     |_ ...
|  ...
|  |_ Webcam
|     |_ 0001.png
|     |_ ...
|_ ...
|_ train_file_1.txt
|_ ...
|_ train_file_N.txt
|_ test_file_1.txt
|_ ...
|_ test_file_N.txt
|_ ...
```

## Dataset
Office-31, Office-home, Bing-Caltech. Please download the dataset first. Then, move the datasets to YOUR_PATH/data/.

## Training

For Office-31 dataset,

Example:

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --noisy_rate 0.4 --noisy_type uniform --dataset Office-31 --source amazon --target dslr --train_epochs 10 --lr 0.001 --loop_prototype 5 --swap_epochs 5 --clean_rate 0.9

```



