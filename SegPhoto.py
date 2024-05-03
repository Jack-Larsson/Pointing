import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sys.path.append("..")

#path to checkpoint
sam_checkpoint = "sam_vit_h_4b8939.pth" #might need to be changed after installing on lab machine
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

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
def segmentFrame(image):
    #create mask
    masks = mask_generator.generate(image)
    
    #not entirely sure what this does until we run it
    print(len(masks))
    print(masks[0].keys())
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show() 

#basically what we want to do after we have the masks
#just an idea, we might not want to return and might just want to display it here

# def pickObject(masks, pointingVector):
#     #go through all the masks
#     for seg in masks:
#         closestDistance = infinity
#         #if the vector intersects this mask, check how far we are from the coordinates that define the segment
#         if pointingVector intersects seg['bbox']:
#             calculate distance from seg['point_coords'] to pointingVector
#             if thisDistance < closestDistance :
#                 targetObject = seg
#     #return the object that we the vector is closest to
#     return targetObject
