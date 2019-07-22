#Code to find the bounding boxes on cuneiform images and return the coordinate list
from __future__ import division
import cv2
import os
import sys

sign_list = ["Dish","One_U","Na","Gar","La","Dab5","Sze3","Ru","Sza","Til","Six_Gesh2","Ke4","Ab","Gal","Ma","Nam","A","Ra","U8","E"]

def draw_rect(image, label, image_id):
	im_height = image.shape[0]
	im_width = image.shape[1]
	coordinate_list = []
	with open(label) as f:
		for i, line in enumerate(f):
			if (line.split(',')[1]) == image_id:
				dim = line.split(',')[0]
				width = int(dim.split('x')[0])
				height = int(dim.split('x')[1])
				line_labels = line.split(',')[3]
				sign_labels = line.split(',')[5]
				sign_labels = sign_labels.split('\n')[0]
				line_labels = line_labels.split('-')
				sign_labels = sign_labels.split('-')
				
				for j in range(0,len(sign_labels)):
					new_line = (sign_labels[j]).split(':')
					sign_class = '_'.join((new_line[0]).split('_')[0:-1])
					if sign_class in sign_list:
						x1 = int((int(new_line[1])/width)*im_width)
						y1 = int((int(new_line[2])/height)*im_height)
						x2 = int(((int(new_line[1]) + int(new_line[3]))/width)*im_width)
						y2 = int(((int(new_line[2]) + int(new_line[4]))/height)*im_height)
						coordinate_list.append([x1,y1,x2,y2])
						cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 1)
				break
	return image, coordinate_list

def main():
	# image_id = "900.png"

	# image = cv2.imread("./images/" + image_id)
	# label = "../templates/annotation.csv"

	# image1 = draw_rect(image, label, image_id)
	# cv2.imwrite('ImageWindow1.png', image1)
	# cv2.waitKey()

if __name__ == "__main__":
	sys.exit(main())