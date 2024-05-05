import cv2
import numpy as np
import findHands
import getSegments
import os

folder = 'ExperimentImages/'

for filename in os.listdir(folder):
    image = cv2.imread(os.path.join(folder,filename))
    print(filename, ' found')
    if image is not None:
        print(filename, ' gave us an image')
        RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        RGB_w_hands = cv2.cvtColor(findHands.drawHands(RGB, image), cv2.COLOR_BGR2RGB)
        mask = getSegments.pickObject(RGB_w_hands )

        color_mask = np.zeros_like(RGB_w_hands )
        color_mask[mask > 0.5] = [230, 144, 255]
        masked_image = cv2.addWeighted(RGB_w_hands , 0.6, color_mask, 0.4, 0)

        cv2.imwrite('MASKED' + filename, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
        








