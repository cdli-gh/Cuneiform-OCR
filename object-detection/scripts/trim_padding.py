#remove excess padding for the synthetic images
from __future__ import division
import os
import sys
import numpy as np
import cv2


def trim_edges(character):
	row_index = []
	col_index = []
	# alternative :np.where(~a.any(axis=1))[0]
	for i in range(0, character.shape[0]):
		row = character[i,:]
		if np.max(row) > 5:
			break
		row_index.append(i)

	for i in xrange(character.shape[0]-1, -1, -1):
		row = character[i,:]
		if np.max(row) > 5:
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
		if np.max(col) > 5:
			break
		col_index.append(j)

	for j in xrange(character.shape[1]-1, -1, -1):
		col = character[:,j]
		if np.max(col) > 5:
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

	for r,d,f in os.walk("./614x907/"):
		for file in f:
			image = cv2.imread("./614x907/" + file)
			image1 = trim_edges(image)
			cv2.imwrite("./images/" + file , image1)
			# cv2.imshow('ImageWindow', image1)
			# cv2.waitKey()

	# image = cv2.imread("./614x907/170.png" )
	# image1 = trim_edges(image)
	# #cv2.imwrite("./images/" + file , image1)
	# cv2.imshow('ImageWindow', image1)
	# cv2.waitKey()

if __name__ == "__main__":
	sys.exit(main())