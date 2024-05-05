import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import pointingVector as pv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sys.path.append("..")

#path to checkpoint
sam_checkpoint = "sam_vit_h_4b8939.pth" #might want to just move into directory but it works
model_type = "vit_h"

#lab machine does not have cuda installed, takes forever to run but scared of messing something up
#ask Dr. Hart before running tests
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam, points_per_side = 16, points_per_batch=32, pred_iou_thresh = 0.92)

#From the image generate a list of all segments. Then search through these segments 
#for the one most likely to be the object being pointing at
#return the mask of the chosen segment
def pickObject(RGBimage):
    #automatically generate masks for entire image
    masks = mask_generator.generate(RGBimage)

    bestScore = 0
    targetObjectMask = masks[0]['segmentation']

    #search through list for the targetObject
    for seg in masks:
        #eliminate objects that are too large to remove background objects such as walls, tables, etc
        if (seg['bbox'][2] / RGBimage.shape[0]) < 0.4 and seg['bbox'][3] / RGBimage.shape[1] < 0.4:
            #if the pointing vector intersects this segment, give it a score
            if pv.boundingBoxIntersect(seg['bbox']):
                score = segScore(seg['bbox'], RGBimage.shape[0] , RGBimage.shape[1])
                #keep track of the best score
                if score > bestScore:
                    targetObjectMask = seg['segmentation']
                    bestScore = score

    #return the mask of the object being pointed at
    return targetObjectMask


#get coordinates for the center of a bounding box
def bboxCenter(bbox):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0] + bbox[2]
    y2 = bbox[1] + bbox[3]

    #average x coordinates and y coordintes
    centerX = (x1 + x2) / 2
    centerY = (y1 + y2) / 2

    #return ordered pair
    return(centerX, centerY)


#get a 'score' for a segment, which is a weighted average of it's
#distance from the pointing vector, distance from the pointing hand, and size
def segScore(bbox, width, height):
    segCoords = bboxCenter(bbox)
    distFromLine = abs(pv.pointLineDistance(segCoords))
    if distFromLine == float('inf'):
        distFromLine = height
    distFromHand = abs(pv.pointToPointDistance(segCoords, (pv.tip_x, pv.tip_y)))
    #favor small items
    #sizeScore = (1 - ((bbox[2] * bbox[3]) / (width * height))) * 0.02
    #favor close items
    proximityScore = (1 - (distFromHand / (width - pv.base_x))) * 0.01
    #favor accuracy most
    accuracyScore = (1 - (distFromLine / height)) * 0.99
    #weighted average determined by testing
    return proximityScore + accuracyScore

    
