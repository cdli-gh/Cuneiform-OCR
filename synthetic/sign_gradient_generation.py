import Augmentor
import cv2
import numpy as np
import os
import sys


#VALID_EXT = [".jpg", ".png"]

# ------------- Morphological operations: -------------

def erode(img, ksize, itr):
	
	kernel = np.ones((ksize,ksize),np.uint8)
	return cv2.erode(img, kernel,iterations = itr)

def skeletonization(img, ksize, img_name):
	
	skel = np.zeros(img.shape,np.uint8)
	size = np.size(img)
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(ksize, ksize))
	done = False
	count = 0

	iter_eroded = []

	while( not done):
	    eroded = cv2.erode(img,element)
	    #cv2.imwrite(str(count) + '.png', eroded)
	    _, eroded_binary = cv2.threshold(eroded,50,255,cv2.THRESH_BINARY)
	    iter_eroded.append(eroded_binary)
	    temp = cv2.dilate(eroded,element)
	    temp = cv2.subtract(img,temp)
	    skel = cv2.bitwise_or(skel,temp)
	    img = eroded.copy()
	    count += 1
	    zeros = size - cv2.countNonZero(img)
	    if zeros==size:
	        done = True

	increment = 255 / count

	add_matrix = np.zeros(img.shape,np.uint8)

	for i in range(0,count):
		if i:
			add_matrix -= (iter_eroded[i]/255) * (increment * (i - 1))
		add_matrix += (iter_eroded[i]/255) * (increment * i)
		#cv2.imwrite( str(i) + 'gradient.png', iter_eroded[i])
	
	#cv2.imwrite('./gradient_signs/gradient_' + img_name, add_matrix)
	return skel

#----output gradient image
def distance_transform(img, img_name, path):
	
	gradient_img = np.zeros(img.shape,np.uint8)
	gradient_img = cv2.distanceTransform(img, cv2.DIST_L2, 5)
	dist2 = cv2.normalize(gradient_img, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)

	#cv2.imwrite(path + "/output/" + img_name, dist2)
	cv2.imwrite(path + "/output/" + img_name, dist2)

def image_to_gradient(path):
	for r, d, f in os.walk(path):
			for directory in d:
				dir_name = str(directory)
				dir_path = path + dir_name
				for r1, d1, f1 in os.walk(dir_path):
					for file in f1:
						image = cv2.imread(os.path.join(dir_path ,file))
						gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
						#ridges_erode_thresh_skel = skeletonization(gray, 3, img)
						distance_transform(gray, file, dir_path)
						#cv2.imwrite('./skeleton_signs/skel_' + img, ridges_erode_thresh_skel)

#-----------------sign generation operations-------------------
#----------making folders for respective signs and moving files into them
def make_folders(path):
	for r, d, f in os.walk(path):
	    for file in f:
	        dir_name = str(file)
	        dir_name1 = dir_name.split(".")[0]
	        os.mkdir(path + dir_name1)
	        old_path = path + dir_name
	        new_path = path + dir_name1 + "/" + dir_name
	        os.rename(old_path, new_path)
	    break

#------------------------augment the sign images
def augment_image(path):
	for r, d, f in os.walk(path):
		for directory in d:
			dir_path = path + str(directory)
			p = Augmentor.Pipeline(dir_path)
			p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=4)
			#p.skew_left_right(probability=1, magnitude=1)
			p.sample(10)
		break

#----------changing path of augmented images from test.py code to individual folder
def change_augment_path(path):
	for r, d, f in os.walk(path):
		for directory in d:
			dir_path = path + str(directory) + "/output"
			for r1, d1, f1 in os.walk(dir_path):
				for file in f1:
					file_name = str(file)
					old_path = path + str(directory) + "/output/" + file_name
					new_path = path + str(directory) + "/" + file_name
					os.rename(old_path, new_path)


#-----------renaming augmented files from random to standard-------------------
def rename_augment(path):
	for r, d, f in os.walk(path):
		for directory in d:
			dir_path = path + str(directory)
			count = 0
			for r1, d1, f1 in os.walk(dir_path):
				for file in f1:
					file_name = str(directory) + "_" + str(count) + ".png"
					count += 1
					old_path = path + str(directory) + "/" + str(file)
					new_path = path + str(directory) + "/" + file_name
					os.rename(old_path, new_path)


#------Code for removing black pixels. Making it transparent-----------
# for r, d, f in os.walk("./input/signs/"):
# 	for directory in d:
# 		path = "./input/signs/" + str(directory)
# 		for r1, d1, f1 in os.walk(path):
# 			for directory1 in d1:
# 				d_path = "./input/signs/" + str(directory) + "/" +  str(directory1)
# 				for r2, d2, f2 in os.walk(d_path):
# 					for file in f2:
# 						file_name = "./input/signs/" + str(directory) + "/" +  str(directory1) + "/" + str(file)
# 						src = cv2.imread(file_name, 1)
# 						tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# 						_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
# 						b, g, r = cv2.split(src)
# 						rgba = [b,g,r, alpha]
# 						dst = cv2.merge(rgba,4)
# 						file_output = "./input/signs/" + str(directory) + "/" + str(file)
# 						cv2.imwrite(file_output, dst)

#------deleting files from sign directory---------

def delete_files(path):
	for r, d, f in os.walk(path):
		for directory in d:
			dir_path = path + str(directory)
			for r1, d1, f1 in os.walk(dir_path):
				for file in f1:
					os.remove(path +  str(directory) + "/" + str(file))	
				break

def main():
	path = sys.argv[1]
	make_folders(path)
	augment_image(path)
	change_augment_path(path)
	rename_augment(path)
	image_to_gradient(path)
	delete_files(path)
	change_augment_path(path)
	
if __name__ == "__main__":
	sys.exit(main())


	