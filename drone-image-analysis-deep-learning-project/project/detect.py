'''
	File name: detect.py
	Program Description: A program that detects and extracts a rice field from a drone image.
	Date: 22 July 2018
	Author: Loria Roie Grace N. Malingan 
'''

from PIL import Image
from datetime import datetime
import os
import random
import sys
import threading
import math
import numpy as np
import cv2
#import segment

def process_img(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting image to grayscale
	blur = cv2.GaussianBlur(gray,(3,3),0) # applying Gaussian blur 
	cv2.imwrite('blur.jpg', blur) # save image
	edged = cv2.Canny(blur, 10, 200) # applying Canny edge detection
	cv2.imwrite('edged.jpg', edged)	# save image
	
	# applying closing function
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
	cv2.imwrite("closed.jpg", closed)
	return closed

def remove_bg(rotated,peri):
	# Isolating the target field from the background
	gray2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
	th, threshed = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
	cnts, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:4]
	idx = 0				
	for c in cnts:
		idx += 1
		if idx == 1:
			ret = crop_cnt(c, rotated)
			break

def rotate_img(field_img):
	# Rotating image
	# rotated = imutils.rotate(field_img,14)
	(h,w) = field_img.shape[:2]
	center = (w/2,h/2)
	M = cv2.getRotationMatrix2D(center, 14, 1.0)
	rotated = cv2.warpAffine(field_img, M, (w, h))
	cv2.imwrite("rotated.jpg", rotated)
	return rotated

def crop_cnt(c, image):
	x,y,w,h = cv2.boundingRect(c)
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	cv2.drawContours(image, [approx], -1, (0,0,0), 3) # drawing contour lines on the image
	field_img = image[y:y+h,x:x+w]	# resize the image based on the detected contour 
	cv2.imwrite("field.tif", field_img) # save the image as a new image
	ret = []
	ret.append(peri)
	ret.append(field_img)
	return ret

def extract_field():
	for infile in os.listdir("./"):
		if infile[-3:] == "tif" and infile != "field.tif":
			print("file : " + infile)
			outfile = infile[:-3] + "jpg"
			im = Image.open(infile) # opens tif file
			print("new filename : " + outfile) # displays new file name
			out = im.convert("RGB") # converts to RGB
			out.save(outfile, quality=100) # saves to outfile

			image = cv2.imread(outfile) # Reads converted image
			closed = process_img(image) # Preprocess the image for better contour detection
			cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # finding contours
			cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:4] # sort the contours ascendingly
			idx = 0
			for c in cnts:
				idx += 1
				if idx == 2:
					ret = crop_cnt(c,image)
					break
			field_img = ret[1] # access the cropped image returned 
			peri = ret[0] # access perimeter of the image returned
			rotated = rotate_img(field_img)
			remove_bg(rotated,peri)

def main():
	extract_field()
if __name__ == "__main__":
	main()
	
