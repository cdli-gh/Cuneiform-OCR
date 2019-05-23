#!/usr/bin/env python

"""
Line detection
"""
from __future__ import division
import numpy 
import cv2 
import matplotlib.pyplot as plt
import sys       # to get command line args
import os
import argparse  # to parse options for us and print a nice help message
import imutils

import torch

from pytorch_hed import Network
from sklearn.cluster import MeanShift, estimate_bandwidth

BW_THRESHOLD = 30
HLP_THRESHOLD = 5
MIN_LINE_LENGTH = 10
MAX_LINE_GAP = 5


assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 41) # requires at least pytorch version 0.4.1

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

#uncomment the statement below if you are utilising a nvidia gpu
#torch.cuda.device(1) # change this if you have a multiple graphics cards and you want to utilize them

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

#finding the obverse and reverse of the rgb cuneiform tablets
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

#resize images to 480x320 for HED 
def resize_image(image):

	height = image.shape[0]
	width = image.shape[1]
	rotate_flag = 0
	#check if we have to rotate image
	if height > width:
		image = imutils.rotate_bound(image, 270)

		height = image.shape[0]
		width = image.shape[1]
		rotate_flag = 1

	ratio = width / height

	#check ratio to decide which direction to add padding
	if ratio <= 1.5:
		height = 320
		width = int(320 * ratio)

		dim = (width, height)
		resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		width_diff = 480 - width
		
		filler = numpy.zeros([320,width_diff,3],dtype=numpy.uint8)
		filler.fill(0) 
		
		final_img = numpy.concatenate((resized, filler), axis=1)

	else:
		height = int(480 / ratio)
		width = 480
		
		dim = (width, height)
		resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		height_diff = 320 - height
		
		filler = numpy.zeros([height_diff,480,3],dtype=numpy.uint8)
		filler.fill(0) 
		
		final_img = numpy.concatenate((resized, filler), axis=0)

	return final_img, rotate_flag

#draw houghlines 
def hough_line(image, original_img, flag):

	def skeletonization(img, ksize):
		skel = numpy.zeros(img.shape,numpy.uint8)
		size = numpy.size(img)
		element = cv2.getStructuringElement(cv2.MORPH_CROSS,(ksize, ksize))
		done = False
		while( not done):
			eroded = cv2.erode(img,element)
			temp = cv2.dilate(eroded,element)
			temp = cv2.subtract(img,temp)
			skel = cv2.bitwise_or(skel,temp)
			img = eroded.copy()
		 
			zeros = size - cv2.countNonZero(img)
			if zeros==size:
				done = True
		return skel

	def merge_lines(x_coord):
		
		X = numpy.array(zip(x_coord,numpy.zeros(len(x_coord))), dtype=numpy.int)
		bandwidth = estimate_bandwidth(X, quantile=0.1)
		ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
		ms.fit(X)
		labels = ms.labels_
		cluster_centers = ms.cluster_centers_

		labels_unique = numpy.unique(labels)
		n_clusters_ = len(labels_unique)

		final_x_coord = []

		for k in range(n_clusters_):
			my_members = labels == k
			final_x_coord.append(int(numpy.mean(X[my_members, 0])))
			#print "cluster {0}: {1}".format(k, X[my_members, 0])
		return final_x_coord

	#edges = cv2.Canny(image,200,250,apertureSize=3)
	thresh, bw_img = cv2.threshold(image, BW_THRESHOLD, 255, cv2.THRESH_BINARY)

	skel_img = skeletonization(bw_img, 3)
	
	if flag:
		lines = cv2.HoughLinesP(image=skel_img,rho=1,theta=numpy.pi, threshold=HLP_THRESHOLD,lines=numpy.array([]), minLineLength=MIN_LINE_LENGTH,maxLineGap=MAX_LINE_GAP)
		num_lines,b,c = lines.shape
		x_coord = []
		for i in range(num_lines):
			x_coord.append(lines[i][0][0])
			x_coord.append(lines[i][0][2])
	
		merged_xcoord = merge_lines(x_coord)

		for i in range(len(merged_xcoord)):
			cv2.line(original_img, (merged_xcoord[i], 0), (merged_xcoord[i], 320), (0, 0, 255), 3, cv2.LINE_AA)

	else:
		lines = cv2.HoughLinesP(image=skel_img,rho=1,theta=numpy.pi/2, threshold=HLP_THRESHOLD,lines=numpy.array([]), minLineLength=MIN_LINE_LENGTH,maxLineGap=MAX_LINE_GAP)

		num_lines,b,c = lines.shape
		y_coord = []
		for i in range(num_lines):
			y_coord.append(lines[i][0][1])
			y_coord.append(lines[i][0][3])
		
		merged_ycoord = merge_lines(y_coord)

		for i in range(len(merged_ycoord)):
			cv2.line(original_img, (0, merged_ycoord[i]), (480, merged_ycoord[i]), (0, 0, 255), 1, cv2.LINE_AA)
	
	return original_img, bw_img, skel_img
	
#function obtained from https://github.com/sniklaus/pytorch-hed/run.py
def estimate(tensorInput, moduleNetwork):
	intWidth = tensorInput.size(2)
	intHeight = tensorInput.size(1)

	assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	#uncomment the command below if you are utilising a nvidia gpu
	#return moduleNetwork(tensorInput.cuda().view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()
	return moduleNetwork(tensorInput.view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()

def line_detection(image, moduleNetwork, filename):
	resized_img, flag = resize_image(image)

	tensorInput = torch.FloatTensor(resized_img[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
		
	tensorOutput = estimate(tensorInput,moduleNetwork)
	
	Hed_image = (tensorOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)

	cv2.imwrite('output/'+'hed'+filename, Hed_image)

	final_image, canny_image, skel_img = hough_line(Hed_image, resized_img, flag)

	cv2.imwrite('output/'+filename,final_image)
	cv2.imwrite('output/'+'c'+filename,canny_image)
	cv2.imwrite('output/'+'skel'+filename,skel_img)
	# plt.imshow(final_image)
	# plt.show()


def main():
	argv = sys.argv

	if "--" not in argv:
		argv = []  # as if no args are passed
	else:
		argv = argv[argv.index("--") + 1:]  # get all args after "--"
	# When --help or no args are given, print this help
	usage_text = (
		"text1"	
		"blender -- [options]"
	)

	parser = argparse.ArgumentParser(description=usage_text)

	parser.add_argument(
		"-i", "--input_img", dest="orginal_img", type=str, required=True,
		help="Input the rgb cuneiform image",
	)
    

	args = parser.parse_args(argv)

	if not argv:
		parser.print_help()
		return

	if (not args.orginal_img):
		print("Error: argument not given, aborting.")
		parser.print_help()
		return
    
	obverse, reverse = obv_rev_segmentation(args.orginal_img)

	#uncomment the command below if you are utilising a nvidia gpu
	#moduleNetwork = Network().cuda().eval()
	
	moduleNetwork = Network().eval()
	line_detection(obverse, moduleNetwork, 'obverse_output.png')
	line_detection(reverse, moduleNetwork, 'reverse_output.png')

	
	
if __name__ == '__main__':
	sys.exit(main())
