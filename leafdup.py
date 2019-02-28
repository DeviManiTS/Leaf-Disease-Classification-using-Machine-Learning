import numpy as np

import argparse

import mahotas

import cv2

import csv

import random
import math
from imutils import paths
def describe(image):

	(means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
	
        colorStats = np.concatenate([means, stds]).flatten()

	

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
        haralick = mahotas.features.haralick(gray).mean(axis=0)

	
  
        return np.hstack([colorStats, haralick])


       
f=open('leaf_features.csv','r+')
writer=csv.writer(f)
ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", required=True, help="path to 8 scene category dataset")
ap.add_argument("-f", "--forest",  help="path to 8 scene category dataset")
args = vars(ap.parse_args())


print("[INFO] extracting features...")

imagePaths = sorted(paths.list_images(args["dataset"]))
labels = []
data = []


for imagePath in imagePaths:
 
	label = imagePath[imagePath.rfind("\\") + 1:].split("_")[0]
 
	image = cv2.imread(imagePath)

	

	features = describe(image)

        if label=='Rust':
         var=0
	elif label=='Mildew':
         var=1
        elif label=='Co':
         var=2
        elif label=='Blight':
         var=3
        features=np.hstack([features,var])
        labels.append(label)
	
        data.append(features)



        writer.writerow(features)
