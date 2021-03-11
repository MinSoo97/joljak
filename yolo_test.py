import cv2
import numpy as np

cap = cv2.VideoCapture(1)

classesFile = 'obj.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

while True:
    succes, img = cap.read()
    
    cv2.imshow('Image' ,img)
    cv2.waitKey(1)