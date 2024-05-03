import cv2
import numpy as np
from pointPicture import *
from SegPhoto import *

image_path = 'test3.jpg'
image = cv2.imread(image_path)
RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

drawHands(RGB, image)
pickObject(RGB)

