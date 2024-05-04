import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import pointingVector as pv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sys.path.append("..")

#path to checkpoint
sam_checkpoint = "/home/bwilab/Downloads/sam_vit_h_4b8939.pth" #might want to just move into directory but it works
model_type = "vit_h"

#lab machine does not have cuda installed, takes forever to run but scared of messing something up
#ask Dr. Hart before running tests
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

def pickObject(RGBimage):
    #go through all the masks
    masks = mask_generator.generate(RGBimage)

    closestDistance = float('inf')
    size = RGBimage.shape[0] * RGBimage.shape[1]
    targetObjectMask = masks[0]['segmentation']

    for seg in masks:
        #if the vector intersects this mask, check how far we are from the coordinates that define the segment
        if (seg['area'] / size) < 0.25 :
            if pv.boundingBoxIntersect(seg['bbox']):
                print('intersecting')
                distance = pv.pointLineDistance(bboxCenter(seg['bbox']))
                if distance < closestDistance:
                    print("we found the best one at", seg['point_coords'])
                    targetObjectMask = seg['segmentation']
                    closestDistance = distance
    #return the object that we the vector is closest to
    return targetObjectMask

#get coordinates for center of a bounding box
def bboxCenter(bbox):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0] + bbox[2]
    y2 = bbox[1] + bbox[3]

    centerX = (x1 + x2) / 2
    centerY = (y1 + y2) / 2

    return(centerX, centerY)
