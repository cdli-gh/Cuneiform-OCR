#crop out the obverse and reverse views from a real RGB cuneiform tablet image
from __future__ import division
import cv2
import os
import sys
import numpy

VALID_EXT = [".jpg", ".png"]


def obv_rev_segmentation(image):
	
	min_h1 = 0
	min_h2 = 0
	
	#find the components of the cuneiform tablet
	def largest_component(image_orig, contours, index):
		offset = 0
		cimg = numpy.zeros_like(image_orig)
		cv2.drawContours(cimg, contours, index, (255,255,255), -1)
		pts = numpy.where(cimg == 255)

		min_h = min(pts[0])
		crop_img = img[min(pts[0])-offset:max(pts[0])+offset, min(pts[1])-offset:max(pts[1])+offset]

		return min_h, crop_img

	img = cv2.imread(image)
	im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	(thresh, bw_img) = cv2.threshold(im_gray, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	contours = cv2.findContours(bw_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]

	contour_index = {}

	for i, contour in enumerate(contours):
		contour_index[i] = cv2.moments(contour)['m00']

	sorted_index = sorted(contour_index, key=contour_index.get, reverse=True)

	min_h1, crop_img_1 = largest_component(img, contours, sorted_index[0])
	min_h2, crop_img_2 = largest_component(img, contours, sorted_index[1])
	
	if min_h1 < min_h2:
		return crop_img_1, crop_img_2
		
	else:
		return crop_img_2, crop_img_1

def get_images_in_directory(path):
	imgs = []
	for f in os.listdir(path):
		f_split  = os.path.splitext(f)
		f_name = f_split[0]
		ext = f_split[1]
		if ext.lower() not in VALID_EXT:
			continue
		imgs.append(f)
	return imgs

def main():
	img_list = get_images_in_directory("./real_images/")
	for img_id in img_list:
		obverse, reverse = obv_rev_segmentation(os.path.join("./real_images/", img_id))
		height = 500
		o_width = int((int(obverse.shape[1])/int(obverse.shape[0]))*500)
		r_width = int((int(reverse.shape[1])/int(reverse.shape[0]))*500)
		resized_obv = cv2.resize(obverse, (o_width, 500)) 
		resized_rev = cv2.resize(reverse, (r_width, 500)) 
		cv2.imwrite("./resized_real_images/" + img_id.split('.')[0] + "_o.jpg", resized_obv)
		cv2.imwrite("./resized_real_images/" + img_id.split('.')[0] + "_r.jpg", resized_rev)


if __name__ == '__main__':
	sys.exit(main())