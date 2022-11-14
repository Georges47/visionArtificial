import mediapipe as mp
import numpy as np
import cv2
import mouse
import tkinter as tk

root_panel = tk.Tk()
screen_size = (root_panel.winfo_screenheight(), root_panel.winfo_screenwidth())
frame_size = (720, 1280)
roi_margin = 0.35
roi_left, roi_top, roi_right, roi_bottom = (int(frame_size[1]*roi_margin), int(frame_size[0]*roi_margin), int(frame_size[1]*(1-roi_margin)), int(frame_size[0]*(1-roi_margin)))

def frameToScreen(frame_pos):
    x,y = screen_size[1]/frame_size[0], screen_size[0]/frame_size[1]    
    screen_pos = [frame_pos[0]*x, frame_pos[1]*y]
    return screen_pos

def roiToFrame(roi_pos):
    x, y = np.clip(roi_pos[0], roi_left, roi_right), np.clip(roi_pos[1], roi_top, roi_bottom)
    frame_pos = [(x-roi_left) * frame_size[0]/(roi_right-roi_left), (y-roi_top) * frame_size[1]/(roi_bottom-roi_top)]
    return frame_pos

def distance_between_points(pt1, pt2):
    d = np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)
    return d

cam = cv2.VideoCapture(0)

drawing_utils = mp.solutions.drawing_utils
hands_utils = mp.solutions.hands

frame_count = 0
debounce = 10
with hands_utils.Hands(static_image_mode=True, max_num_hands = 1, min_detection_confidence=0.5) as hands:
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            continue
        
        frame = cv2.resize(frame, (frame_size[1], frame_size[0]))
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        hands_processed = hands.process(rgb_frame)
        
        if hands_processed.multi_hand_landmarks:
            for landmarks in hands_processed.multi_hand_landmarks:
                index_tip = drawing_utils._normalized_to_pixel_coordinates(
                    landmarks.landmark[hands_utils.HandLandmark.INDEX_FINGER_TIP].x,
                    landmarks.landmark[hands_utils.HandLandmark.INDEX_FINGER_TIP].y,
                    frame_size[1], frame_size[0])                
                                
                thumb_tip = drawing_utils._normalized_to_pixel_coordinates(
                    landmarks.landmark[hands_utils.HandLandmark.THUMB_TIP].x, 
                    landmarks.landmark[hands_utils.HandLandmark.THUMB_TIP].y, 
                    frame_size[1], frame_size[0])
                
                middle_tip = drawing_utils._normalized_to_pixel_coordinates(
                    landmarks.landmark[hands_utils.HandLandmark.MIDDLE_FINGER_TIP].x, 
                    landmarks.landmark[hands_utils.HandLandmark.MIDDLE_FINGER_TIP].y, 
                    frame_size[1], frame_size[0])
                           
                if frame_count == debounce:
                    if thumb_tip is not None and middle_tip is not None:
                        if distance_between_points(thumb_tip, middle_tip)<60:
                            mouse.click()
                            print("clicked")
                        else:
                            mouse.release()
                    frame_count = 0

                screen_pos = frameToScreen(roiToFrame(index_tip))
                mouse.move(screen_pos[0], screen_pos[1])
                
                frame_count += 1
                drawing_utils.draw_landmarks(frame, landmarks, hands_utils.HAND_CONNECTIONS)

        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (255, 0, 0), 1)        
        cv2.imshow("TP4CameraCursor", frame)
        if cv2.waitKey(1)&0xFF == 27:
            break
cam.release()
cv2.destroyAllWindows()
