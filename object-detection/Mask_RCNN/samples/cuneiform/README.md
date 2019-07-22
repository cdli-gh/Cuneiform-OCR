# Detection and Classification Example:

<img src="https://i.imgur.com/Yaqhdca.jpg" width="300" height="401.5"> <img src="https://i.imgur.com/evjfuBq.jpg" width="300" height="401.5"> <img src="https://i.imgur.com/6yPoT2B.jpg" width="300" height="401.5"> <img src="https://i.imgur.com/OXlHkaM.jpg" width="300" height="401.5">

* Currently there are a lot of false positives for characters like 1(disz), 1(u), a. This is due to only 20 signs being used for training. As we increase the number of signs for training it should learn to distinguish between 2(disz) vs 2x 1(disz) ie; some of these characters form a part of other complex characters. It has also detected certain complex characters like gal, na, la if the corresponding real image had a moderate-to-good quality image of the signs. 

* Another case was in some real images that had a lot of 1(disz) characters, only 3-4 1(disz) were detected. This is due to how I built the synthetic dataset. My dataset currently has a roughly equal distribution of 78 signs compared to a real dataset where some signs 1(disz), a appear more than others similarly to how a,e,i have a higher distribution than x,y,q in English. Therefore the model learned that at max a cuneiform image will have 3-4 of a particular sign and hence failed to detect all of them. A possible solution is to use disentanglement where it won’t learn such features.

## Installation
From the [Dataset page](https://drive.google.com/open?id=1g4JeaJrmQ8K_fh_TOYeAyd6s6grqH2Dr) page:
1. Download `mask_rcnn_object_0186.h5`. Save it in the root directory of the repo (the `mask_rcnn` directory).
2. Download `cuneiform_dataset.zip`. It contains the train, validation, test dataset. Expand it such that it's in the path `mask_rcnn/input_images/`.

## Train the Cuneiform model

```
python3 cuneiform.py train --dataset=/path/to/balloon/dataset --weights=/path/to/weight_file
```

The code in `cuneiform.py` is set to train for 10K steps (1000 epochs of 100 steps each), and using a batch size of 2. 
Update the schedule to fit your needs.
