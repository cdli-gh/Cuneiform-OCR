#!/usr/bin/env python
"""
Output a JSON file that contains the coordinates of the cuneiform sign masks. This file is needed for Mask RCNN.
"""
from __future__ import division
import cv2
import os
import sys
import numpy as np
import argparse  # to parse options for us and print a nice help message

#JSON template for Mask RCNN
JSON_TEMPLATE = "\"%s\":{\"fileref\":\"\",\"size\":%d,\"filename\":\"%s\",\"base64_img_data\":\"\",\"file_attributes\":{},\"regions\":{%s}}"

#Template that contains the mask coordinates and respective class name
REGION_TEMPLATE = "\"%s\": {\"shape_attributes\":{\"name\":\"polygon\",\"all_points_x\":[%s],\"all_points_y\":[%s]},\"region_attributes\":{\"name\":\"%s\"}}"

#trim sign images for building mask 
def trim_edges(character):
	row_index = []
	col_index = []
	# alternative :np.where(~a.any(axis=1))[0]
	for i in range(0, character.shape[0]):
		row = character[i,:]
		if np.max(row) > 0:
			break
		row_index.append(i)

	for i in xrange(character.shape[0]-1, -1, -1):
		row = character[i,:]
		if np.max(row) > 0:
			break
		row_index.append(i)

	del_count = 0
	row_index = sorted(row_index)
	#deleting the rows(top/bottom)
	for row in row_index:
		character = np.delete(character, (row - del_count), axis=0)
		del_count += 1

	for j in range(0, character.shape[1]):
		col = character[:,j]
		if np.max(col) > 0:
			break
		col_index.append(j)
	#should be j for new set of synthetic images
	for i in xrange(character.shape[1]-1, -1, -1):
		col = character[:,j]
		if np.max(col) > 0:
			break
		col_index.append(j)

	del_count = 0
	col_index = sorted(col_index)
	#deleting the columns(sides)
	for col in col_index:
		character = np.delete(character, (col - del_count), axis=1)
		del_count += 1

	return character

def main():

	argv = sys.argv

	if "--" not in argv:
		argv = []  # as if no args are passed
	else:
		argv = argv[argv.index("--") + 1:]  # get all args after "--"
	# When --help or no args are given, print this help
	usage_text = (
		"python find_mask.py -- [options]"
	)

	parser = argparse.ArgumentParser(description=usage_text)

	parser.add_argument(
		"-isyn", "--input_syndir", dest="synth_path", type=str, required=True,
		help="Input the synthetic image directory",
	)
	parser.add_argument(
		"-isign", "--input_signdir", dest="sign_path", type=str, required=True,
		help="Input the sign image directory",
	)
	parser.add_argument(
		"-ianota", "--input_anotafile", dest="anota_path", type=str, required=True,
		help="Input the annotation file path",
	)
    
	args = parser.parse_args(argv)

	if not argv:
		parser.print_help()
		return

	if (not args.synth_path or
		not args.sign_path or
		not args.anota_path):
		print("Error: argument not given, aborting.")
		parser.print_help()
		return

	sign_list = ["Dish","One_U","Na","Gar","La","Dab5","Sze3","Ru","Sza","Til","Six_Gesh2","Ke4","Ab","Gal","Ma","Nam","A","Ra","U8","E"]
	json = "{"

	file = open("./via_region_data.json","w+")
	count = 0
	with open(args.anota_path) as f:
		for i, line in enumerate(f):
			new_line = line.split(',')[5]
			new_line1 = (new_line.split('\n')[0]).split('-')
			file_name_0 = (line.split(',')[1]).split('.')[0]
			file_name = line.split(',')[1]
			#change check for training, validation and testing respectively
			if int(file_name_0) >= 1510:
				if os.path.isfile(os.path.join(args.synth_path, file_name)):
					#read synthetic image
					synthetic_image = cv2.imread(os.path.join(args.synth_path, file_name))
					#get dimensions of synthetic image
					template_dimension = (line.split(',')[0]).split('x')
					width = int(template_dimension[0])
					height = int(template_dimension[1])
					file_size = os.stat(os.path.join(args.synth_path, file_name)).st_size
					regions = ""
					region_id = 0

					for j in range(0,len(new_line1)):
						new_line2 = (new_line1[j]).split(':')
						sign_file_name = new_line2[0]
						sign_class = '_'.join((new_line2[0]).split('_')[0:-1])
						#check if the sign belongs to the 20 class sign list
						if sign_class in sign_list:
							sign_image_path = os.path.join(args.sign_path, sign_class, sign_file_name)
							sign_image = cv2.imread(sign_image_path)
							trimmed_sign_image = trim_edges(sign_image)
							#get dimensions of sign image
							dim = (int(new_line2[3]), int(new_line2[4]))
							resized_sign_image = cv2.resize(trimmed_sign_image, dim, interpolation = cv2.INTER_AREA)
							#get left, top coordinates of sign position in synthetic image
							x_coord = int(new_line2[1])
							y_coord = int(new_line2[2])
							im_bw = cv2.cvtColor(resized_sign_image, cv2.COLOR_RGB2GRAY)
							ret,thresh = cv2.threshold(im_bw,0,255,0)
							#find contours of respective sign image
							contours = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
							polygon_x = []
							polygon_y = []
							#for all the contours of a respective sign, append the boundary coordinates
							for cnt in contours:
								epsilon = 0.01*cv2.arcLength(cnt,True)
								approx = cv2.approxPolyDP(cnt,epsilon,True)
								for i_1 in range(0, len(approx)):
									for j_1 in range(0, len(approx[i_1])):
										approx[i_1][j_1][0] = (approx[i_1][j_1][0]) + x_coord
										approx[i_1][j_1][0] = int(((approx[i_1][j_1][0])/width)*synthetic_image.shape[1])
										polygon_x.append(approx[i_1][j_1][0])
										approx[i_1][j_1][1] = (approx[i_1][j_1][1]) + y_coord
										approx[i_1][j_1][1] = int(((approx[i_1][j_1][1])/height)*synthetic_image.shape[0])
										polygon_y.append(approx[i_1][j_1][1])
							if region_id:
				 				regions += ","	
				 			regions += REGION_TEMPLATE % (str(region_id), ','.join(map(str,polygon_x)), ','.join(map(str,polygon_y)), sign_class)
				 			region_id += 1
				 	if count:
				 		json += ","
				 	json += JSON_TEMPLATE % (file_name+str(file_size), file_size, file_name, regions)
			 	
			 	count += 1

			#change check for training, validation and testing respectively
		 	elif file_name == "2000.png":
		 		break
	json += "}"	 		
	file.write(json)
 	file.close()
 	
if __name__ == "__main__":
	sys.exit(main())