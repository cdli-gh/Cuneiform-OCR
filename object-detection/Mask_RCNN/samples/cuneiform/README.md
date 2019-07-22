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
