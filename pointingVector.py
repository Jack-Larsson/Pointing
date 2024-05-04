import numpy as np

base_x = 0
base_y = 0
extended_tip_x = 0
extended_tip_y = 0
tip_x = 0
tip_y = 0

#find distance from a point to a line
#used to determine which segment we are pointing at
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

    if (tip_x < x2 and px < tip_x) or (tip_x > x2 and px > tip_x):
        return float('inf')

    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

    return num / den


#given the bounding box of a segment, return true if our pointing vector passes through it
def boundingBoxIntersect(bounding_box) -> bool:
    global base_x
    global base_y 
    global extended_tip_x
    global extended_tip_y
    #print (base_x, base_y, extended_tip_x, extended_tip_y)
    slope = (extended_tip_y - base_y) / (extended_tip_x - base_x)

    topLeftX = bounding_box[0]
    topLeftY = bounding_box[1]
    botRightX = bounding_box[0] + bounding_box[2]
    botRightY = bounding_box[1] + bounding_box[3]

    leftLineY = (slope * (topLeftX - base_x)) + base_y
    rightLineY = (slope * (botRightX - base_x)) + base_y

    if (leftLineY < topLeftY and rightLineY < topLeftY) or (leftLineY > botRightY and rightLineY > botRightY):
        return False

    return True