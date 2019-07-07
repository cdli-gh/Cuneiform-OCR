#draw bounding boxes over the signs and lines
from __future__ import division
import cv2
import os
import sys


def draw_rect(image, label, image_id):
	im_height = image.shape[0]
	im_width = image.shape[1]
	with open(label) as f:
		for i, line in enumerate(f):

			if (line.split(',')[1]) == image_id:
				dim = line.split(',')[0]
				width = dim.split('x')[0]
				height = dim.split('x')[1]
				line_labels = line.split(',')[3]
				line_labels = line_labels.split('-')
				for j in range(0,len(line_labels)):
					new_line = (line_labels[j]).split(':')
					x1 = int(new_line[0])
					y1 = int((int(new_line[1])/int(height))*int(im_height))
					x2 = int(new_line[0]) + int(im_width)
					y2 = int(new_line[1]) + int(new_line[3])
					y2 = int((y2/int(height))*int(im_height))
					cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
				break

	return image

def main():
	image_id = "900.png"

	image = cv2.imread("./images/" + image_id)
	label = "../templates/annotation.csv"

	image1 = draw_rect(image, label, image_id)
	cv2.imshow('ImageWindow', image1)
	cv2.waitKey()

if __name__ == "__main__":
	sys.exit(main())