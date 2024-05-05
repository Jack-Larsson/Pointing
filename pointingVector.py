import numpy as np

base_x = 0
base_y = 0
extended_tip_x = 0
extended_tip_y = 0
tip_x = 0
tip_y = 0

#pre: drawHands() has been called
#find distance from a point to the pointing vector
#used to help determine which segment we are pointing at
def pointLineDistance(point):
    global base_x
    global base_y 
    global extended_tip_x
    global extended_tip_y
    px, py = point
    x1 = base_x
    y1 = base_y
    x2 = extended_tip_x
    y2 = extended_tip_y

    #if the point is behind the pointing hand, it is certainly not the object being pointed at
    if (tip_x < x2 and px < (tip_x * 1.1)) or (tip_x > x2 and px > (tip_x * 0.9)):
        return float('inf')

    #distance between a line and a point formula
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

    return num / den


#given the bounding box of a segment, return true if our pointing vector passes through it
def boundingBoxIntersect(bounding_box) -> bool:
    global tip_x
    global tip_y 
    global extended_tip_x
    global extended_tip_y

    #calculate slope of the vector in the frame
    slope = (extended_tip_y - tip_y) / (extended_tip_x - tip_x)
    
    #get corners of bounding box
    topLeftX = bounding_box[0]
    topLeftY = bounding_box[1]
    botRightX = bounding_box[0] + bounding_box[2]
    botRightY = bounding_box[1] + bounding_box[3]

    #get y coordinate of pointing vector at the left and right ends of the bounding box
    leftLineY = (slope * (topLeftX - tip_x)) + tip_y
    rightLineY = (slope * (botRightX - tip_x)) + tip_y

    #if the line is above the box on both sides or below the box on both sides, it never passed through the box
    if (leftLineY < topLeftY and rightLineY < topLeftY) or (leftLineY > botRightY and rightLineY > botRightY):
        return False
    #otherwise, the line passed through the box
    return True

#get distance between two points in the image
def pointToPointDistance(point1, point2):
    p1x, p1y = point1
    p2x, p2y = point2
    #distance between two points formula
    sqrDiffX = (p2x - p1x)**2 
    sqrDiffY = (p2y - p1y)**2 
    dist = np.sqrt(sqrDiffX + sqrDiffY)
    return dist