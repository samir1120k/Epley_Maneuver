import streamlit as st
import numpy as np
import mediapipe as mp
import cv2
import time
import pyttsx3

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3,model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

#initialize camara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    st.error("Error: Could not open camera.")
    st.stop()

#initialize text to speech
engine=pyttsx3.init()
engine.setProperty('rate',150)
# Streamlit UI
st.title("AI Based BPPV Manuver Guider")
st.image("diagnosis.gif", caption="Diagnosis",use_container_width=True)
st.image("bppv_treatment.gif", caption="steps in BPPV", use_container_width=True)
frame_placeholder = st.empty()

def speak(text):
    """Speak the given text using pyttsx3."""
    engine.say(text)
    engine.runAndWait()

def get_head_angles(landmarks_object): # Renamed parameter for clarity
    """Calculate 3D head angles: yaw, pitch, roll."""
    # Access the actual list of landmarks HERE
    lm = landmarks_object.landmark

    # Now use 'lm' for indexing
    nose = lm[mp_pose.PoseLandmark.NOSE]
    left_ear = lm[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR]
    left_eye = lm[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = lm[mp_pose.PoseLandmark.RIGHT_EYE]

    # --- Calculations remain the same ---
    mid_ear = np.array([(left_ear.x + right_ear.x) / 2,
                        (left_ear.y + right_ear.y) / 2,
                        (left_ear.z + right_ear.z) / 2]) 
    nose_vec = np.array([nose.x, nose.y, nose.z]) - mid_ear
    # atan2(delta_x, delta_z) is generally better for yaw
    yaw = np.degrees(np.arctan2(nose_vec[0], nose_vec[2]))

    eye_mid = np.array([(left_eye.x + right_eye.x) / 2,
                        (left_eye.y + right_eye.y) / 2,
                        (left_eye.z + right_eye.z) / 2]) 
    nose_to_eye = np.array([nose.x, nose.y, nose.z]) - eye_mid
    # atan2(-delta_y, sqrt(delta_x^2 + delta_z^2)) is often used for pitch
    pitch = np.degrees(np.arctan2(-nose_to_eye[1], np.sqrt(nose_to_eye[0]**2 + nose_to_eye[2]**2)))

    ear_vec = np.array([right_ear.x - left_ear.x, right_ear.y - left_ear.y, right_ear.z - left_ear.z])
    # atan2(delta_y, delta_x) for roll
    roll = np.degrees(np.arctan2(ear_vec[1], ear_vec[0]))

    return yaw, pitch, roll

def get_body_angle(landmarks):
    """Calculate torso angle (yaw-like rotation relative to camera)."""
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    mid_shoulder = np.array([left_shoulder.x + right_shoulder.x, left_shoulder.y + right_shoulder.y,
                                 left_shoulder.z + right_shoulder.z]) / 2
    mid_hip = np.array([left_hip.x + right_hip.x, left_hip.y + right_hip.y, left_hip.z + right_hip.z]) / 2
    torso_vec = mid_shoulder - mid_hip
    body_yaw = np.degrees(np.arctan2(torso_vec[0], torso_vec[2]))
    return body_yaw

def get_leg_spine_angle(landmarks):
    """Calculate the angle between the spine (torso) and the left leg."""
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

    spine_vec = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z]) - \
                  np.array([left_hip.x, left_hip.y, left_hip.z])
    leg_vec = np.array([left_knee.x, left_knee.y, left_knee.z]) - \
                np.array([left_hip.x, left_hip.y, left_hip.z])
    
    cosine_angle = np.dot(spine_vec, leg_vec) / (np.linalg.norm(spine_vec) * np.linalg.norm(leg_vec))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def get_hip_roll(landmarks):
    """Calculate the roll angle of the hips."""
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    
    hip_vec = np.array([right_hip.x - left_hip.x, right_hip.y - left_hip.y, right_hip.z - left_hip.z])
    roll = np.degrees(np.arctan2(hip_vec[1], hip_vec[0]))
    return roll

def is_full_body_visible(landmarks, frame_width, frame_height):
    """Check if the full body is visible in the frame."""
    key_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE
    ]
    
    for landmark in key_landmarks:
        lm = landmarks[landmark]
        # Check if landmark is within frame bounds and has sufficient visibility
        if (lm.visibility < 0.5 or 
            lm.x < 0.05 or lm.x > 0.95 or 
            lm.y < 0.05 or lm.y > 0.95):
            return False
    return True
#Bakwas app
# Main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        st.error("Error: Could not read frame from camera.")
        break

    # Flip the frame horizontally for a selfie-view display.
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_time = time.time()

    # Process the image and get the pose landmarks.
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        frame_height, frame_width = frame.shape[:2]
        
        if not is_full_body_visible(landmarks, frame_width, frame_height):
                    cv2.putText(frame, "Please come in front of the camera", 
                                (frame_width // 4, frame_height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

        elif results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, # Draw on the BGR frame
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
            head_yaw, head_pitch, head_roll = get_head_angles(results.pose_landmarks)
            body_yaw = get_body_angle(landmarks)
            leg_spine_angle = get_leg_spine_angle(landmarks)
            hip_roll = get_hip_roll(landmarks)
            if head_yaw is not None:
                cv2.putText(frame, f"Yaw: {head_yaw:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(frame, f"Pitch: {head_pitch:.1f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,1.5, (255, 255,255), 3)
                cv2.putText(frame, f"Roll: {head_roll:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX,1.5, (255, 255, 255), 3)
                cv2.putText(frame, f"body Yaw: {body_yaw:.1f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(frame, f"leg spine: {leg_spine_angle:.1f}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX,1.5, (255, 255,255), 3)
                cv2.putText(frame, f"body Roll: {hip_roll:.1f}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX,1.5, (255, 255,255), 3)
                
    # Display the frame in the Streamlit UI.
    frame_placeholder.image(frame, channels="BGR", use_container_width=True)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows.
cap.release()
cv2.destroyAllWindows()



