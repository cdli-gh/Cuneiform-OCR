# Line Detection



## Pre-requisites

All the pre-requisite libraries can be found in requirements.txt

To download the pre-trained models for performing [HED(Holistically-Nested Edge Detection)](https://arxiv.org/abs/1504.06375), 
run `bash download.bash`. 

## Usage

To perform line detection:

```
python line_detection.py -- --i <cueniform image directory path>
```

<cueniform image directory path> contains the rgb images of cuneiform tablets.

## Expected Output
img id = **P124776**

<img src="https://i.imgur.com/xCrjoml.jpg" width="200" height="401.5"> <img src="https://i.imgur.com/8ITicJk.png" height="240" width="160">
<img src="https://i.imgur.com/AGL0GsG.png" height="240" width="160">


## References
```
[1]  @inproceedings{Xie_ICCV_2015,
         author = {Saining Xie and Zhuowen Tu},
         title = {Holistically-Nested Edge Detection},
         booktitle = {IEEE International Conference on Computer Vision},
         year = {2015}
     }
```

```
[2]  @misc{pytorch-hed,
         author = {Simon Niklaus},
         title = {A Reimplementation of {HED} Using {PyTorch}},
         year = {2018},
         howpublished = {\url{https://github.com/sniklaus/pytorch-hed}}
    }
```
