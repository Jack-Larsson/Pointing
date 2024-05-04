import cv2
import numpy as np
import findHands
import getSegments

image_path = 'images/test3.jpg'
image = cv2.imread(image_path)
RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

findHands.drawHands(RGB, image)
mask = getSegments.pickObject(RGB)

color_mask = np.zeros_like(RGB)
color_mask[mask > 0.5] = [230, 144, 255]
masked_image = cv2.addWeighted(RGB, 0.6, color_mask, 0.4, 0)

cv2.imwrite('highlight_object.png', cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))




