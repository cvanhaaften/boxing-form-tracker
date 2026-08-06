import cv2
import mediapipe as mp
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_CONNECTIONS = frozenset([
    # face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
])

model_path = 'pose_landmarker.task'
latest_result = None

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def angle_between(a, b, c):
    # b is the vertex
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_poses=1, #change to 6?
    result_callback=print_result)

#Open the webcamera
cap = cv2.VideoCapture(0)

# Punch position
reset = False

with PoseLandmarker.create_from_options(options) as landmarker:
  punch_count = 0
  
  while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.imshow('frame', frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    #pass mp_image to landmarker and convert
    timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms)

    if latest_result and latest_result.pose_landmarks:
        landmarks = latest_result.pose_landmarks[0]

        # Print lines between the landmarks
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            start_point = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
            end_point = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))

            cv2.line(frame, start_point, end_point, (255, 255, 255), 2)

        # Print landmarks
        for landmark in latest_result.pose_landmarks[0]:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        #Essential landmarks for punch tracking
        left_wrist = latest_result.pose_landmarks[0][15]
        left_elbow = latest_result.pose_landmarks[0][13]
        left_shoulder = latest_result.pose_landmarks[0][11]
        right_wrist = latest_result.pose_landmarks[0][16]
        right_elbow = latest_result.pose_landmarks[0][14]
        right_shoulder = latest_result.pose_landmarks[0][12]
        nose = latest_result.pose_landmarks[0][0]
        mouth = latest_result.pose_landmarks[0][10]

        # Get arm angle
        right_angle = angle_between((right_wrist.x, right_wrist.y), (right_elbow.x, right_elbow.y),
         (right_shoulder.x, right_shoulder.y))
        left_angle = angle_between((left_wrist.x, left_wrist.y), (left_elbow.x, left_elbow.y),
         (left_shoulder.x, left_shoulder.y))
        

        # Check if back to starting position
        if(left_angle < 45 and right_angle < 45 and left_wrist.x > nose.x and right_wrist.x < nose.x
            and left_wrist.y - right_wrist.y < 0.1):
            reset = True
        
        # Check for straight punch
        elif(((left_angle > 140) or (right_angle > 140)) and reset):
            punch_count += 1
            reset = False
       
        # Check for hook
        elif((left_wrist.x < nose.x or right_wrist.x > nose.x) 
            and (left_elbow.y - left_wrist.y < 0.1 or right_elbow.y - right_wrist.y > 0.1) and reset):
            punch_count += 1
            reset = False
        
        # Check for uppercut (needs fixing is kind of buggy)
        elif((left_wrist.y < mouth.y or right_wrist.y < mouth.y) and reset):
            punch_count += 1
            reset = False

        



    # Print punch count
    position = (100, 300)
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 10
    color = (255, 255, 255)
    thickness = 6
    cv2.putText(frame, str(punch_count), position, font, fontScale, color, thickness)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
      break

  cap.release() 
  cv2.destroyAllWindows()