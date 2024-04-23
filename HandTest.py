import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
def drawHands(azureCap):
    #turn raw buffer from KinectTest into cv::Mat
    img_np = np.frombuffer(azureCap, dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, 
                        min_detection_confidence=0.5) as hands:
        
        # Read image, flip it around y-axis for correct handedness output 
        image = cv2.flip(img, 1)
        
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()

    #I don't think this is necessary I think it's just for output
        # for hand_landmarks in results.multi_hand_landmarks:
        #     print('hand_landmarks:', hand_landmarks)
        #     print(
        #         f'Index finger tip coordinates: (',
        #         f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
        #         f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
        #     )

        # List to store index finger joints coordinates
        index_finger_joints = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            for joint in mp_hands.HandLandmark:
                # Collect index finger landmarks
                if (joint == mp_hands.HandLandmark.INDEX_FINGER_TIP) or (joint == mp_hands.HandLandmark.INDEX_FINGER_MCP):
                    index_finger_joints.append((int(hand_landmarks.landmark[joint].x), 
                    int(hand_landmarks.landmark[joint].y), int(hand_landmarks.landmark[joint].z)))
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        
        # cv2.imwrite(
        #     '/tmp/annotated_image.png', cv2.flip(annotated_image, 1))
        # # Draw hand world landmarks.
        # for hand_world_landmarks in results.multi_hand_world_landmarks:
        #     mp_drawing.plot_landmarks(
        #         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
                       
    return index_finger_joints       

