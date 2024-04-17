# Introduction
Source code of the MVSEN.
## Requirements
* Python==3.7.0
* pytorch==1.7.0
* torchvision==0.8.0
* torchaudio==0.7.0
* pytorch-pretrained-bert==0.6.2
  
## Pretrained model
If you don't want to train from scratch, you can download the pre-trained model from [here](https://drive.google.com/drive/folders/122NdXKb16Trxx7cmoI5Orb2VnKnR4F6t?usp=drive_link) (for Flickr30K)
```bash
i2t: 503.0
Image to text: 77.8  93.3  97.2
Text to image: 59.2  85.0  90.5
t2i: 497.9
Image to text: 79.5  94.6 97.7
Text to image: 61.2  86.6  91.7
```
## Download Data 
We utilize the image feature created by SCAN, downloaded [here](https://github.com/kuanghuei/SCAN). Some related text data can be found [here](https://drive.google.com/drive/folders/1y55ccAlmoT7VSnNzLBYLPI-oYNRKX--K?usp=drive_link).

```
# Train on Flickr30K
python train.py --batch_size 64 --data_path data/ --dataset f30k --Matching_direction t2i --num_epochs 30
python train.py --batch_size 64 --data_path data/ --dataset f30k --Matching_direction i2t --num_epochs 30

## Evaluation
Run ```test.py``` to evaluate the trained models on f30k.
```
