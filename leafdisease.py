
from __future__ import print_function

from sklearn.cross_validation import train_test_split

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from imutils import paths

import numpy as np

import argparse

import mahotas

import cv2


def describe(image):

	(means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
	
        colorStats = np.concatenate([means, stds]).flatten()

	

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
        haralick = mahotas.features.haralick(gray).mean(axis=0)

	
  
        return np.hstack([colorStats, haralick])


       

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
	
        labels.append(label)
	
        data.append(features)



(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data),
np.array(labels), test_size=0.25, random_state=42)

model = DecisionTreeClassifier(random_state=84)
if args["forest"] > 0: 
  model = RandomForestClassifier(n_estimators=20, random_state=42)


print("[INFO] training model...")

model.fit(trainData, trainLabels)



predictions = model.predict(testData)

print(classification_report(testLabels, predictions))

for i in np.random.randint(0, high=len(imagePaths), size=(10,)):

     imagePath = imagePaths[i]
	
     filename = imagePath[imagePath.rfind("\\") + 1:]

     image = cv2.imread(imagePath)
	
     features = describe(image)
	
     prediction = model.predict(features.reshape(1, -1))[0]


     print("[PREDICTION] {}: {}".format(filename, prediction))

     cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
	
     cv2.imshow("Image", image)
	
     cv2.waitKey(0)
