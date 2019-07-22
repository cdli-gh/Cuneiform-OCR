# Mask R-CNN for Object Detection and Segmentation of Cuneiform signs

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. It was implemented by https://github.com/matterport and modified here for Cuneiform sign detection. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

### More information regarding this implementation can be found here: https://github.com/matterport/Mask_RCNN/

# Getting Started
* [cuneiform.py](samples/cuneiform/cuneiform.py) is for setting up the training process.

* [final_test.py](mrcnn/final_test.py) is for testing on synthetic cuneiform images.

* [real_test.py](mrcnn/real_test.py) is for testing on real cuneiform images.

* ([model.py](mrcnn/model.py), [utils.py](mrcnn/utils.py), [config.py](mrcnn/config.py)): These files contain the main Mask RCNN implementation. 

# Training on synthetic Cuneiform Images

```
# To train a model
python3 samples/cuneiform/cuneiform.py train --dataset=/path/to/cuneiform_images/ --model=/path/to/weights
```
The training schedule, learning rate, and other parameters should be set in `samples/cuneiform/cuneiform.py`.

# Test on real Cuneiform Images

```
python mrcnn/real_test.py -- -isynth /path/to/test images/ -iweight /path/to/weight_file -iresults /path/to/results_folder 
```
# Results

The model has trained on only 20 signs (a, ab, gal, gar, la, 1(u), 1(disz), sze3, sza, ru, ra etc.

<img src="https://i.imgur.com/Yaqhdca.jpg" width="300" height="401.5"> <img src="https://i.imgur.com/evjfuBq.jpg" width="300" height="401.5"> <img src="https://i.imgur.com/6yPoT2B.jpg" width="300" height="401.5"> <img src="https://i.imgur.com/OXlHkaM.jpg" width="300" height="401.5">

* Currently there are a lot of false positives for characters like 1(disz), 1(u), a. This is due to only 20 signs being used for training. As we increase the number of signs for training it should learn to distinguish between 2(disz) vs 2x 1(disz) ie; some of these characters form a part of other complex characters. It has also detected certain complex characters like gal, na, la if the corresponding real image had a moderate-to-good quality image of the signs. 

* Another case was in some real images that had a lot of 1(disz) characters, only 3-4 1(disz) were detected. This is due to how I built the synthetic dataset. My dataset currently has a roughly equal distribution of 78 signs compared to a real dataset where some signs 1(disz), a appear more than others similarly to how a,e,i have a higher distribution than x,y,q in English. Therefore the model learned that at max a cuneiform image will have 3-4 of a particular sign and hence failed to detect all of them. A possible solution is to use disentanglement where it won’t learn such features.


## Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `requirements.txt`.

## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).

4. Download final trained weights (mask_rcnn_object_0186.h5) from here.

## Citation
```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```
