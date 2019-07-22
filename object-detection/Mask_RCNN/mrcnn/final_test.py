""" 
  Testing code for synthetic test images
  Usage: python final_test.py -- -isynth /path/to/test images/ -ijson /path/to/test json file
        -iweight /path/to/weight_file -ianota /path/to/annotation file 
"""
from __future__ import division
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import cv2
import os
import sys
import json
import csv
import argparse
from draw_rect import draw_rect

RESULTS_PATH = "./test_results/"

SIGN_DICT_REV = {"1":"Dish", "2":"One_U", "3":"Na","4":"Gar","5":"La","6":"Dab5","7":"Sze3","8":"Ru","9":"Sza","10":"Til","11":"Six_Gesh2","12":"Ke4","13":"Ab","14":"Gal","15":"Ma","16":"Nam","17":"A","18":"Ra","19":"U8","20":"E"}

SIGN_DICT = {"Dish":1,"One_U":2,"Na":3,"Gar":4,"La":5,"Dab5":6,"Sze3":7,"Ru":8,"Sza":9,"Til":10,"Six_Gesh2":11,"Ke4":12,"Ab":13,"Gal":14,"Ma":15,"Nam":16,"A":17,"Ra":18,"U8":19,"E":20}

#calculate intersection of ground truth box and predicted box
def bb_intersection_over_union(gt_box, x1, y1, x2, y2):
     # determine the (x, y)-coordinates of the intersection rectangle
     SI= max(0, min(gt_box[2], x2) - max(gt_box[0], x1)) * max(0, min(gt_box[3],y2) - max(gt_box[1], y1))
     SG = (gt_box[2] - gt_box[0])*(gt_box[3] - gt_box[1])
     SP = (x2-x1)*(y2-y1)
     SU = SG + SP - SI
     overlap_ratio = SI/SU
     return overlap_ratio

# draw an image with detected objects
def draw_image_with_boxes(filename, boxes_list, class_num, class_score, gt_signs, boxes_gt, file_name):
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
     gt_sign_count = 0
     for gt_box in boxes_gt:
          p_count = 0
          for box in boxes_list:
               # get coordinates
               y1, x1, y2, x2 = box
               overlap_ratio = bb_intersection_over_union(gt_box, x1, y1, x2, y2)
               # calculate width and height of the box
               width, height = x2 - x1, y2 - y1
               # create the shape
               rect = Rectangle((x1, y1), width, height, fill=False, color='red')
               # draw the box
               font = cv2.FONT_HERSHEY_PLAIN
               sign_name = str(class_num[p_count])
               sign_name = str(SIGN_DICT_REV[sign_name])
               #change overlap ratio of ground truth and predicted box, currently 50%
               if overlap_ratio >= 0.5:
                    if gt_signs[gt_sign_count] == sign_name:
                         cv2.putText(syn_image,sign_name,(x1,y2), font, 1,(0,255,0),2,cv2.LINE_AA)
                         cv2.rectangle(syn_image, (x1, y1), (x2, y2), (0,255,0), 1)
                    else:
                         cv2.putText(syn_image,sign_name,(x1,y2), font, 1,(0,0,255),2,cv2.LINE_AA)
                         cv2.rectangle(syn_image, (x1, y1), (x2, y2), (0,0,255), 1)
               ax.add_patch(rect)
               predicted_box.append([x1,y1,x2,y2])
               p_count += 1
          
          gt_sign_count += 1
     #cv2.imwrite(os.path.join(RESULTS_PATH, file_name), syn_image)
     # show the plot
     #pyplot.show() 
      
def generate_ground_truth_txt(filename, gt_signs, gt_box):
     gt_file = open("./ground-truth/" + filename + ".txt", "w")
     count = 0
     for sign in gt_signs:
          to_write = sign + " " + str(gt_box[count][0]) + " " + str(gt_box[count][1]) + " " + str(gt_box[count][2]) + " " + str(gt_box[count][3])
          gt_file.write(to_write)
          gt_file.write("\n")
          count += 1
     gt_file.close()

def generate_predicted_txt(filename, boxes_list, class_num, class_score):
     p_file = open("./detection-results/" + filename + ".txt", "w")
     p_count = 0
     for box in boxes_list:
          # get coordinates
          y1, x1, y2, x2 = box
          sign_name = str(class_num[p_count])
          sign_name = str(SIGN_DICT_REV[sign_name])
          score = class_score[p_count]
          to_write = sign_name + " " + str(score) + " " + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2)
          p_file.write(to_write)
          p_file.write("\n")
          p_count += 1
     p_file.close()

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
          "python synth_test.py -- [options]"
     )

     parser = argparse.ArgumentParser(description=usage_text)

     parser.add_argument(
          "-isynth", "--input_syndir", dest="synth_path", type=str, required=True,
          help="Input the synthetic image directory",
     )
     parser.add_argument(
          "-ijson", "--input_jsonfile", dest="json_path", type=str, required=True,
          help="Input the annotation file",
     )
     parser.add_argument(
          "-iweight", "--input_weightfile", dest="weight_path", type=str, required=True,
          help="Input the weight file",
     )
     parser.add_argument(
          "-ianota", "--input_anotafile", dest="anota_path", type=str, required=True,
          help="Input the annotation file",
     )
    
     args = parser.parse_args(argv)

     if not argv:
          parser.print_help()
          return

     if (not args.synth_path or
          not arg.json_path or 
          not args.weight_path or
          not args.anota_path):
          print("Error: argument not given, aborting.")
          parser.print_help()
          return

     #change model path here
     rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
     #load weight to the model
     rcnn.load_weights(args.weight_path, by_name=True)
     for r,d,f in os.walk(args.synth_path):
          for file in f:
               annotations = json.load(open(os.path.join(args.json_path, "via_region_data.json")))

               filename = file
               filesize = os.stat(os.path.join(args.synth_path, filename)).st_size
               key = filename + str(filesize)
               
               the_file = open(args.anota_path, 'r')
               reader = csv.reader(the_file)
               N = int(filename.split('.')[0])
               line = next((x for i, x in enumerate(reader) if i == N), None)
               the_file.close()

               image = cv2.imread(os.path.join(args.synth_path, filename))
               #get coordinates of signs for ground truth txt preparation
               marked_image, coord_list = draw_rect(image, args.anota_path, filename)
               
               ground_truth_signs = []
               for region_num in sorted(annotations[key]['regions']):
                    #print("Sign_name: %s, Id: %d" % 
                    #      (annotations[key]['regions'][region_num]['region_attributes']['name'], 
                    #       SIGN_DICT[annotations[key]['regions'][region_num]['region_attributes']['name']]))
                    ground_truth_signs.append(annotations[key]['regions'][region_num]['region_attributes']['name'])

               # load photograph
               img = load_img(os.path.join(args.synth_path, filename))
               img = img_to_array(img)
               # make prediction
               results = rcnn.detect([img], verbose=0)
               # visualize the results
               r = results[0]
               #draw_image_with_boxes(os.path.join(args.synth_path, filename), r['rois'], r['class_ids'], r['scores'], ground_truth_signs, coord_list, filename)
               file_name = filename.split('.')[0]
               #generate detected data txt files for mAP calculation
               generate_predicted_txt(file_name, r['rois'], r['class_ids'], r['scores'])
               #generate ground truth txt files for mAP calculation
               generate_ground_truth_txt(file_name, ground_truth_signs, coord_list)

if __name__ == "__main__":
     sys.exit(main())
