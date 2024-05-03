import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from pointPicture import *
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sys.path.append("..")

#path to checkpoint
sam_checkpoint = "/home/bwilab/Downloads/sam_vit_h_4b8939.pth" #might need to be changed after installing on lab machine
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

# image = cv2.imread('test.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#from the tutorial I don't think we really need this
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

#image needs to be RGB, which MediaPipe also uses so shouldn't be an issue

#basically what we want to do after we have the masks
#just an idea, we might not want to return and might just want to display it here

def pickObject(RGBimage):
    #go through all the masks
    masks = mask_generator.generate(RGBimage)
    print(len(masks))
    print(masks[0].keys())
    closestDistance = float('inf')
    size = RGBimage.shape[0] * RGBimage.shape[1]
    for seg in masks:
        #if the vector intersects this mask, check how far we are from the coordinates that define the segment
        if (seg['area'] / size) < 0.25 :
            if boundingBoxIntersect(seg['bbox']):
                if pointLineDistance(seg['point_coords'][0]) < closestDistance :
                    print("we found the best one at", seg['point_coords'])
    #return the object that we the vector is closest to
    print("done with loop")
    plt.figure(figsize=(20,20))
    plt.imshow(RGBimage)
    show_anns(masks)
    plt.axis('off')
    plt.show()
    print("completed")

#   ax = plt.gca()
#     ax.set_autoscale_on(False)
#     polygons = []
#     color = []
#     print(targetObject['point_coords'])
#     m = targetObject['segmentation']
#     img = np.ones((m.shape[0], m.shape[1], 3))
#     color_mask = np.random.random((1, 3)).tolist()[0]
#     for i in range(3):
#         img[:,:,i] = color_mask[i]
#     ax.imshow(np.dstack((img, m*0.35)))