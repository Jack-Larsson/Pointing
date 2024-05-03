import cv2
import mediapipe as mp
import numpy as np

global base_x
global base_y 
global extended_tip_x
global extended_tip_y

base_x = 0
base_y  = 0
extended_tip_x = 1
extended_tip_y = 1

def pointLineDistance(point):
    px, py = point
    x1 = base_x
    y1 = base_y
    x2 = extended_tip_x
    y2 = extended_tip_y
    print(x1, y1, x2, y2)
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    print(den)
    return num / den


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.05)


def drawHands(RGB, image):
    result = hand.process(RGB)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # setup index finger coords
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]

            tip_x, tip_y = int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0])
            # line start
            base_x = int(index_base.x * image.shape[1]) 
            base_y = int(index_base.y * image.shape[0])

            pip_x, pip_y = int(index_pip.x * image.shape[1]), int(index_pip.y * image.shape[0])
            dip_x, dip_y = int(index_dip.x * image.shape[1]), int(index_dip.y * image.shape[0])
            
        # draw a line from index base to tip
        #cv2.line(image, (base_x, base_y), (tip_x, tip_y), (0, 255, 0), 4)
        
        # if the middle joints in finger aren't close enough to the line don't draw the extended ray
        distance_threshold = 14  
        pip_distance = pointLineDistance((pip_x, pip_y))
        dip_distance = pointLineDistance((dip_x, dip_y))

        #if pip_distance < distance_threshold and dip_distance < distance_threshold:
        # draw the ray
        direction = np.array([tip_x - base_x, tip_y - base_y])
        norm_direction = direction / np.linalg.norm(direction)

        # line end
        extended_tip_x = int(tip_x + norm_direction[0] * 1000)  
        extended_tip_y = int(tip_y + norm_direction[1] * 1000)

        #cv2.line(image, (tip_x, tip_y), (extended_tip_x, extended_tip_y), (255, 0, 0), 4)
    
        
        #mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    #cv2.imshow("hand", image)

def boundingBoxIntersect(bounding_box) -> bool:
    lsX = base_x
    lsY = base_y
    leX = extended_tip_x
    leY = extended_tip_y
    slope = (leY - lsX) / (leX - lsX)

    topLeftX = bounding_box[0]
    topLeftY = bounding_box[1]
    botRightX = bounding_box[0] + bounding_box[2]
    botRightY = bounding_box[1] + bounding_box[3]

    leftLineY = slope * (topLeftX + lsX) - lsY
    rightLineY = slope * (botRightX + lsX) - lsY

    if (leftLineY < topLeftY and rightLineY < topLeftY) or (leftLineY > botRightY and rightLineY > botRightY):
        return False

    return True