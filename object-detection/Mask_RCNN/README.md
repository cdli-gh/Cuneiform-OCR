# Mask R-CNN for Object Detection and Segmentation of Cuneiform signs

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. It was implemented by https://github.com/matterport and modified here for Cuneiform sign detection. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

### More information regarding this implementation can be found here: https://github.com/matterport/Mask_RCNN/

# Getting Started
* [cuneiform.py](samples/cuneiform/cuneiform.py) is for setting up the training process.

* [final_test.py](mrcnn/final_test.py) is for testing on synthetic cuneiform images.

* [real_test.py](mrcnn/real_test.py) is for testing on real cuneiform images.

* ([model.py](mrcnn/model.py), [utils.py](mrcnn/utils.py), [config.py](mrcnn/config.py)): These files contain the main Mask RCNN implementation. 

# Training on Cuneiform Images

```
# To train a model
python3 samples/cuneiform/cuneiform.py train --dataset=/path/to/cuneiform_images/ --model=/path/to/weights
```
The training schedule, learning rate, and other parameters should be set in `samples/cuneiform/cuneiform.py`.

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
