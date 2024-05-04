import cv2
import mediapipe as mp
import numpy as np
import pointingVector as pv

def drawHands(RGB, image):
    #process image and find hands
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hand = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.05)
    result = hand.process(RGB)

    #get relevant coordinates for index finger
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # setup index finger coords
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]

            pv.tip_x = int(index_tip.x * image.shape[1]) 
            pv.tip_y = int(index_tip.y * image.shape[0])
            #start of pointing vector
            pv.base_x = int(index_base.x * image.shape[1]) 
            pv.base_y = int(index_base.y * image.shape[0])

            # pip_x, pip_y = int(index_pip.x * image.shape[1]), int(index_pip.y * image.shape[0])
            # dip_x, dip_y = int(index_dip.x * image.shape[1]), int(index_dip.y * image.shape[0])
            
        # # if the middle joints in finger aren't close enough to the line don't draw the extended ray
        # distance_threshold = 20  
        # pip_distance = pv.point_line_distance((pip_x, pip_y))
        # dip_distance = pv.point_line_distance((dip_x, dip_y))

        # if pip_distance < distance_threshold and dip_distance < distance_threshold:

        #make ray
        direction = np.array([pv.tip_x - pv.base_x, pv.tip_y - pv.base_y])
        norm_direction = direction / np.linalg.norm(direction)

        # line end
        pv.extended_tip_x = int(pv.tip_x + norm_direction[0] * 1000)  
        pv.extended_tip_y = int(pv.tip_y + norm_direction[1] * 1000)

        #testing
        cv2.line(image, (pv.tip_x, pv.tip_y), (pv.extended_tip_x, pv.extended_tip_y), (255, 0, 0), 4)

        #draw hands on image
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
