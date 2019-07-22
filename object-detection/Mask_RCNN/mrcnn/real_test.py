#Code to test on real cuneiform images
from __future__ import division
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from final_test import generate_predicted_txt
import cv2
import os
import sys
import json
import csv
import argparse

SIGN_DICT_REV = {"1":"Dish", "2":"One_U", "3":"Na","4":"Gar","5":"La","6":"Dab5","7":"Sze3","8":"Ru","9":"Sza","10":"Til","11":"Six_Gesh2","12":"Ke4","13":"Ab","14":"Gal","15":"Ma","16":"Nam","17":"A","18":"Ra","19":"U8","20":"E"}

SIGN_DICT = {"Dish":1,"One_U":2,"Na":3,"Gar":4,"La":5,"Dab5":6,"Sze3":7,"Ru":8,"Sza":9,"Til":10,"Six_Gesh2":11,"Ke4":12,"Ab":13,"Gal":14,"Ma":15,"Nam":16,"A":17,"Ra":18,"U8":19,"E":20}

# draw an image with detected objects
def draw_image_with_boxes(filename, boxes_list, class_num, class_score, file_name, results_path):
     # load the image
     data = pyplot.imread(filename)
     syn_image = cv2.imread(filename)
     # plot the image
     pyplot.imshow(data)
     # get the context for drawing boxes
     ax = pyplot.gca()
     # for num in class_num:
     #     num = str(num)
     #     print(SIGN_DICT_REV[num])
     # print(class_num)
     # print(class_score)
     # plot each box
     predicted_box = []
     p_count = 0
     for box in boxes_list:
          # get coordinates
          y1, x1, y2, x2 = box
          # calculate width and height of the box
          width, height = x2 - x1, y2 - y1
          # create the shape
          rect = Rectangle((x1, y1), width, height, fill=False, color='red')
          # draw the box
          font = cv2.FONT_HERSHEY_PLAIN
          sign_name = str(class_num[p_count])
          sign_name = str(SIGN_DICT_REV[sign_name])
          cv2.putText(syn_image,sign_name,(x1,y2), font, 1,(0,255,0),2,cv2.LINE_AA)
          cv2.rectangle(syn_image, (x1, y1), (x2, y2), (0,255,0), 1)
          ax.add_patch(rect)
          predicted_box.append([x1,y1,x2,y2])
          p_count += 1
     
     cv2.imwrite(os.path.join(results_path, file_name), syn_image)
     #show the plot
     #pyplot.show() 

# define the test configuration
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 20
 
def main():
     argv = sys.argv

     if "--" not in argv:
          argv = []  # as if no args are passed
     else:
          argv = argv[argv.index("--") + 1:]  # get all args after "--"
     # When --help or no args are given, print this help
     usage_text = (
          "python real_test.py -- [options]"
     )

     parser = argparse.ArgumentParser(description=usage_text)

     parser.add_argument(
          "-isynth", "--input_syndir", dest="synth_path", type=str, required=True,
          help="Input the synthetic image directory",
     )
     parser.add_argument(
          "-iweight", "--input_weightfile", dest="weight_path", type=str, required=True,
          help="Input the weight file",
     )
     parser.add_argument(
          "-iresults", "--input_resultsdir", dest="results_path", type=str, required=True,
          help="Input the results directory",
     )
    
     args = parser.parse_args(argv)

     if not argv:
          parser.print_help()
          return

     if (not args.synth_path or
         not args.weight_path or
         not args.results_path):
         print("Error: argument not given, aborting.")
         parser.print_help()
         return

     rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
     rcnn.load_weights(args.weight_path, by_name=True)
     for r,d,f in os.walk(args.synth_path):
          for file in f:
               filename = file
               #load image
               img = load_img(os.path.join(args.synth_path, filename))
               img = img_to_array(img)
               # make prediction
               results = rcnn.detect([img], verbose=0)
               # visualize the results
               r = results[0]
               draw_image_with_boxes(os.path.join(args.synth_path, filename), r['rois'], r['class_ids'], r['scores'], filename, args.results_path)
               file_name = filename.split('.')[0]
               #generate detected data txt files for mAP calculation
               generate_predicted_txt(file_name, r['rois'], r['class_ids'], r['scores'])

if __name__ == "__main__":
     sys.exit(main())
