import numpy as np

#find distance from a point to a line
#used to determine if we are pointing and which segment we are pointing at
def point_line_distance(point):
    global base_x
    global base_y 
    global extended_tip_x
    global extended_tip_y

    px, py = point
    x1 = base_x
    y1 = base_y
    x2 = extended_tip_x
    y2 = extended_tip_y

    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

    return num / den


#given the bounding box of a segment, return true if our pointing vector passes through it
def boundingBoxIntersect(bounding_box) -> bool:
    global base_x
    global base_y 
    global extended_tip_x
    global extended_tip_y
    slope = (base_x - base_y) / (extended_tip_x - extended_tip_y)

    topLeftX = bounding_box[0]
    topLeftY = bounding_box[1]
    botRightX = bounding_box[0] + bounding_box[2]
    botRightY = bounding_box[1] + bounding_box[3]

    leftLineY = slope * (topLeftX + base_x) - base_y
    rightLineY = slope * (botRightX + base_x) - base_y

    if (leftLineY < topLeftY and rightLineY < topLeftY) or (leftLineY > botRightY and rightLineY > botRightY):
        return False

    return True