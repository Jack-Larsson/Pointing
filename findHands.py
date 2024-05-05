import cv2
import mediapipe as mp
import numpy as np
import pointingVector as pv

def drawHands(RGB, image):
    #process image and find hands
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hand = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.05)
    result = hand.process(RGB)

    #get relevant coordinates for index finger
    if result.multi_hand_landmarks:
        straightest = 0
        for hand_landmarks in result.multi_hand_landmarks:
            # get relevant coords for this hand
            this_index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            this_index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            this_index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            this_index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]

            pip_x, pip_y = int(this_index_pip.x * image.shape[1]), int(this_index_pip.y * image.shape[0])
            tip_x, tip_y = int(this_index_tip.x * image.shape[1]), int(this_index_tip.y * image.shape[0])
            base_x, base_y = int(this_index_base.x * image.shape[1]), int(this_index_base.y * image.shape[0])
            dip_x, dip_y = int(this_index_dip.x * image.shape[1]), int(this_index_dip.y * image.shape[0])

            baseTipDist = pv.pointToPointDistance((base_x, base_y), (tip_x, tip_y))
            basePipDist = pv.pointToPointDistance((pip_x, pip_y), (base_x, base_y))
            dipPipDist = pv.pointToPointDistance((pip_x, pip_y), (dip_x, dip_y))
            dipTipDist = pv.pointToPointDistance((dip_x, dip_y), (tip_x, tip_y))
            straightness = baseTipDist / (basePipDist + dipPipDist + dipTipDist)
            # try and get the straightest pointer finger in the photo
            if (straightness > straightest):
                straightest = straightness
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]


        pv.tip_x = int(index_tip.x * image.shape[1]) 
        pv.tip_y = int(index_tip.y * image.shape[0])
        #start of pointing vector
        pv.base_x = int(index_base.x * image.shape[1]) 
        pv.base_y = int(index_base.y * image.shape[0])

        #make ray
        direction = np.array([pv.tip_x - pv.base_x, pv.tip_y - pv.base_y])
        norm_direction = direction / np.linalg.norm(direction)

        # line end
        pv.extended_tip_x = int(pv.tip_x + norm_direction[0] * 1000)  
        pv.extended_tip_y = int(pv.tip_y + norm_direction[1] * 1000)

        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.line(image, (pv.tip_x, pv.tip_y), (pv.extended_tip_x, pv.extended_tip_y), (255, 0, 0), 4)
    return image
