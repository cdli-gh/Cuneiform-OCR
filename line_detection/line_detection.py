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

from run import Network


assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 41) # requires at least pytorch version 0.4.1

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

#uncomment the command below if you are utiling a nvidia gpu
#torch.cuda.device(1) # change this if you have a multiple graphics cards and you want to utilize them

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

#finding the obverse and reverse of the rgb cuneiform tablets
def obv_rev_segmentation(image):
	
	min_h1 = 0
	min_h2 = 0
	
	def largest_component(image_1, contours_1, index):
		offset = 0
		cimg = numpy.zeros_like(image_1)
		cv2.drawContours(cimg, contours_1, index, (255,255,255), -1)
		pts = numpy.where(cimg == 255)

		min_h = min(pts[0])
		crop_img = img[min(pts[0])-offset:max(pts[0])+offset, min(pts[1])-offset:max(pts[1])+offset]

		return min_h, crop_img

	img = cv2.imread(image)
	im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	(thresh, im_bw) = cv2.threshold(im_gray, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	contours = cv2.findContours(im_bw, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]

	contour_index = {}

	for i, contour in enumerate(contours):
		contour_index[i] = cv2.moments(contour)['m00']

	sorted_index = sorted(contour_index, key=contour_index.get, reverse=True)

	min_h1, crop_img_1 = largest_component(img, contours, sorted_index[0])
	min_h2, crop_img_2 = largest_component(img, contours, sorted_index[1])
	
	if min_h1 < min_h2:
		return crop_img_1, crop_img_2
		
	if min_h1 > min_h2:
		return crop_img_2, crop_img_1

#resize images to 480x320 for HED 
def resize_image(image):

	height = image.shape[0]
	width = image.shape[1]
	#check if we have to rotate image, ie; width > height
	if height > width:
		rotated = imutils.rotate_bound(image, 270)

		height = rotated.shape[0]
		width = rotated.shape[1]

	ratio = width / height

	#check ratio to decide 
	if ratio <= 1.5:
		height = 320
		width = int(320 * ratio)

		dim = (width, height)
		resized = cv2.resize(rotated, dim, interpolation = cv2.INTER_AREA)
		width_diff = 480 - width
		
		filler = numpy.zeros([320,width_diff,3],dtype=numpy.uint8)
		filler.fill(0) 
		
		final_img = numpy.concatenate((resized, filler), axis=1)

	else:
		height = int(480 * ratio)
		width = 480
		
		dim = (width, height)
		resized = cv2.resize(rotated, dim, interpolation = cv2.INTER_AREA)
		height_diff = 480 - height
		
		filler = numpy.zeros([height_diff,480,3],dtype=numpy.uint8)
		filler.fill(0) 
		#filler[:] = 0
		
		final_img = numpy.concatenate((resized, filler), axis=0)

	return final_img

#output the lines
def hough_line(image, original_img):

	edges = cv2.Canny(image,50,150,apertureSize = 3)

	lines = cv2.HoughLinesP(image=edges,rho=1,theta=numpy.pi/180, threshold=50,lines=numpy.array([]), minLineLength=100,maxLineGap=80)

	a,b,c = lines.shape
	for i in range(a):
		cv2.line(original_img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
	
	return original_img
	
#function obtained from https://github.com/sniklaus/pytorch-hed/run.py
def estimate(tensorInput, moduleNetwork):
	intWidth = tensorInput.size(2)
	intHeight = tensorInput.size(1)

	assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	#uncomment the command below if you are utiling a nvidia gpu
	#return moduleNetwork(tensorInput.cuda().view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()
	return moduleNetwork(tensorInput.view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()

def line_detection(image, moduleNetwork, filename):
	resized_img = resize_image(image)

	tensorInput = torch.FloatTensor(resized_img[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
		
	tensorOutput = estimate(tensorInput,moduleNetwork)
	
	Hed_image = (tensorOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)

	final_image = hough_line(Hed_image, resized_img)

	cv2.imwrite(filename,final_image)
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

	#uncomment the command below if you are utilizing a nvidia gpu
	#moduleNetwork = Network().cuda().eval()
	
	moduleNetwork = Network().eval()
	line_detection(obverse, moduleNetwork, 'obverse_output.png')
	line_detection(reverse, moduleNetwork, 'reverse_output.png')

	
	
if __name__ == '__main__':
	sys.exit(main())
