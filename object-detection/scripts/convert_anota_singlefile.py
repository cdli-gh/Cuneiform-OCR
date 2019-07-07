#converting annotation file to individual txt files corresponding to the synthetic images 
#using darknet format
from __future__ import division
import os
import cv2

#find the count of each sign across the whole dataset
sign_dict = {}
with open("annotation.csv") as f:
	for i, line in enumerate(f):
		new_line = line.split(',')[5]
		new_line1 = (new_line.split('\n')[0]).split('-')
		file_name = (line.split(',')[1]).split('.')[0]
		towrite = str(len(new_line1))
		for j in range(0,len(new_line1)):
			new_line2 = (new_line1[j]).split(':')
			sign_class = '_'.join((new_line2[0]).split('_')[0:-1])
			if sign_class in sign_dict:
				sign_dict[sign_class] += 1
			elif sign_class not in sign_dict:
				sign_dict[sign_class] = 1

#choose the top 20 numerous signs
sign_list = []
for i in range(0,20):
	sign_list.append((sorted(sign_dict.items(), key=lambda x: x[1], reverse=True)[i])[0])

sign_dict = {}
count = 0
for sign in sign_list:
	sign_dict[sign] = count
	count += 1

i = 0

#generate annotation files in darknet format
with open("annotation.csv") as f:
	for i, line in enumerate(f):
		new_line = line.split(',')[5]
		new_line1 = (new_line.split('\n')[0]).split('-')
		file_name = (line.split(',')[1]).split('.')[0]
		file_name = "labels/sign_original/" + file_name + ".txt"
		file = open(file_name,"w+")
		file_name1 = (line.split(',')[1])
		path = "./images/" + file_name1
		image = cv2.imread(path)
		for j in range(0,len(new_line1)):
			new_line2 = (new_line1[j]).split(':')
			sign_class = '_'.join((new_line2[0]).split('_')[0:-1])
			if sign_class in sign_list:
				#x_center & y_center of bounding rect over sign normalised with template dimensions
				x_center_norm = (int(new_line2[1]) + int(new_line2[3])/2)/int(image.shape[1])
				y_center_norm = (int(new_line2[2]) + int(new_line2[4])/2)/int(image.shape[0])
				#width & height of sign image normalised with template dimensions
				sign_width_norm = int(new_line2[3])/int(image.shape[1])
				sign_height_norm = int(new_line2[4])/int(image.shape[0])
				final_line = str(sign_dict[sign_class]) + " " + str(x_center_norm) + " " + str(y_center_norm) + " " + str(sign_width_norm) + " " + str(sign_height_norm)
				file.write(final_line)
	 			file.write("\n")
 		file.close()
		

class_file = open("class_number.txt","w+")
for x in sign_dict:
	line_format = x + ":" + str(sign_dict[x])
	class_file.write(line_format)
	class_file.write("\n")
class_file.close()
