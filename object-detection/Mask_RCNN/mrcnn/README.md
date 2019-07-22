## Test the Cuneiform Model

For synthetic images:
```
python final_test.py -- -isynth /path/to/test images/ -ijson /path/to/test_json_file -iweight /path/to/weight_file -ianota /path/to/annotation_file 
```
For real cuneiform images:
```
python real_test.py -- -isynth /path/to/test images/ -iweight /path/to/weight_file -iresults /path/to/results_folder 
```
* For synthetic images: ground-truth and detected sign coordinates in txt files are generated. These are used for mAP calculation.
  [mAP Tool](https://github.com/Cartucho/mAP) was used to calculate mAP, loss avg for each class and also draw boundings boxes on the images.

* For real images: detected sign coordinates in txt files are generated. For ground-truth coordinates in txt files, corresponding signs should be
  labelled on real images using [BBox Label tool](https://github.com/puzzledqs/BBox-Label-Tool). We can then use mAP tool to find mAP, loss avg.
  
<img src="https://i.imgur.com/GUxi8eY.jpg" width="300" height="401.5"> <img src="https://i.imgur.com/hwTFYrb.jpg" width="300" height="401.5"> <img src="https://i.imgur.com/FY7pKtj.jpg" width="300" height="401.5">
