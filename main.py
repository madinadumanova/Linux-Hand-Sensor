
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import pyautogui
import math
import os

# --- 1. Linux & Safety Settings ---
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0 
os.environ['DISPLAY'] = ':0' 

# State & Config
prev_x, prev_y = 0, 0
smoothing = 5 
click_threshold = 0.05
is_clicking = False 
model_path = 'hand_landmarker.task'

# Drawing Helpers
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands_connections

# --- 2. Setup MediaPipe ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO)

with vision.HandLandmarker.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)
    screen_w, screen_h = pyautogui.size()

    print("Air Mouse mit Vektor-Visualisierung gestartet...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = detector.detect_for_video(mp_image, timestamp)
        
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # --- VEKTOREN ZEICHNEN ---
                # Wandelt die Landmarks in ein Format um, das MediaPipe zeichnen kann
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in hand_landmarks
                ])
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks_proto, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2), # Punkte
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2) # Linien/Vektoren
                )

                # --- LOGIK ---
                index_tip = hand_landmarks[8]
                thumb_tip = hand_landmarks[4]

                # Cursor-Vektor berechnen
                target_x = index_tip.x * screen_w
                target_y = index_tip.y * screen_h
                curr_x = prev_x + (target_x - prev_x) / smoothing
                curr_y = prev_y + (target_y - prev_y) / smoothing
                
                pyautogui.moveTo(int(curr_x), int(curr_y))
                prev_x, prev_y = curr_x, curr_y

                # Distanz-Vektor (Pinch) visualisieren
                dist = math.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
                
                # Linie zwischen Daumen und Zeigefinger zeichnen
                line_color = (0, 255, 0) if dist < click_threshold else (0, 0, 255)
                cv2.line(frame, 
                         (int(thumb_tip.x * w), int(thumb_tip.y * h)), 
                         (int(index_tip.x * w), int(index_tip.y * h)), 
                         line_color, 3)

                if dist < click_threshold:
                    if not is_clicking:
                        pyautogui.click()
                        is_clicking = True 
                else:
                    is_clicking = False

        cv2.imshow('Air Mouse Vektor-Ansicht', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
