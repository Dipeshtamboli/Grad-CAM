import openslide 
from PIL import Image
import os
from os import walk
folder_add = "/home/dipesh/grad_cam/data/train"

files = os.listdir(folder_add)

for j in files:
	# print j
	cwd = os.getcwd()
	file2_add=folder_add +"/"+str(j)
	# print file2_add 
	file2=os.listdir(file2_add)
	for k in file2:
		# print k
		img=Image.open(file2_add + "/" + k)
		x = img.size
		if not x==(256,256):
			print(file2_add + "/" + k, x)
