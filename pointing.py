import cv2
import numpy as np
import findHands
import getSegments
import os

folder = 'ExperimentImages/'

#process all images in experiment folder
for filename in os.listdir(folder):
    image = cv2.imread(os.path.join(folder,filename))
    if image is not None:
        #get RGB photo for segment-anything and MediaPipe
        RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #get RGB Image with hands and vector drawn on it
        RGB_w_vector = cv2.cvtColor(findHands.drawHands(RGB, image), cv2.COLOR_BGR2RGB)
        mask = getSegments.pickObject(RGB_w_vector)
        #highlight the segment we choose
        color_mask = np.zeros_like(RGB_w_vector)
        color_mask[mask > 0.5] = [230, 144, 255]
        masked_image = cv2.addWeighted(RGB_w_vector , 0.6, color_mask, 0.4, 0)
        #save image with highlighted segment
        cv2.imwrite('MASKED' + filename, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
        








