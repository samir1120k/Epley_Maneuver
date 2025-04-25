import cv2
import streamlit as st
import numpy as np
import mediapipe as mp
import av
import pyttsx3
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def get_head_angles(landmarks):
    lm = landmarks.landmark
    nose = lm[mp_pose.PoseLandmark.NOSE]
    left_ear = lm[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR]
    left_eye = lm[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = lm[mp_pose.PoseLandmark.RIGHT_EYE]

    mid_ear = np.array([(left_ear.x + right_ear.x) / 2, (left_ear.y + right_ear.y) / 2, (left_ear.z + right_ear.z) / 2]) 
    nose_vec = np.array([nose.x, nose.y, nose.z]) - mid_ear
    yaw = np.degrees(np.arctan2(nose_vec[0], nose_vec[2]))

    eye_mid = np.array([(left_eye.x + right_eye.x) / 2, (left_eye.y + right_eye.y) / 2, (left_eye.z + right_eye.z) / 2]) 
    nose_to_eye = np.array([nose.x, nose.y, nose.z]) - eye_mid
    pitch = np.degrees(np.arctan2(-nose_to_eye[1], np.sqrt(nose_to_eye[0]**2 + nose_to_eye[2]**2)))

    ear_vec = np.array([right_ear.x - left_ear.x, right_ear.y - left_ear.y, right_ear.z - left_ear.z])
    roll = np.degrees(np.arctan2(ear_vec[1], ear_vec[0]))

    return yaw, pitch, roll

def get_body_angle(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    mid_shoulder = np.array([left_shoulder.x + right_shoulder.x, left_shoulder.y + right_shoulder.y, left_shoulder.z + right_shoulder.z]) / 2
    mid_hip = np.array([left_hip.x + right_hip.x, left_hip.y + right_hip.y, left_hip.z + right_hip.z]) / 2
    torso_vec = mid_shoulder - mid_hip
    return np.degrees(np.arctan2(torso_vec[0], torso_vec[2]))

def get_leg_spine_angle(landmarks):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    spine_vec = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z]) - np.array([left_hip.x, left_hip.y, left_hip.z])
    leg_vec = np.array([left_knee.x, left_knee.y, left_knee.z]) - np.array([left_hip.x, left_hip.y, left_hip.z])
    cosine_angle = np.dot(spine_vec, leg_vec) / (np.linalg.norm(spine_vec) * np.linalg.norm(leg_vec))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def get_hip_roll(landmarks):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    hip_vec = np.array([right_hip.x - left_hip.x, right_hip.y - left_hip.y, right_hip.z - left_hip.z])
    return np.degrees(np.arctan2(hip_vec[1], hip_vec[0]))

class PoseAnalyzer(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            yaw, pitch, roll = get_head_angles(results.pose_landmarks)
            body_yaw = get_body_angle(landmarks)
            leg_angle = get_leg_spine_angle(landmarks)
            hip_roll = get_hip_roll(landmarks)

            cv2.putText(image, f"Yaw: {yaw:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(image, f"Pitch: {pitch:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(image, f"Roll: {roll:.1f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(image, f"Body Yaw: {body_yaw:.1f}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            cv2.putText(image, f"Leg-Spine Angle: {leg_angle:.1f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
            cv2.putText(image, f"Hip Roll: {hip_roll:.1f}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        return image

# Streamlit UI
st.title("🧠 AI-Based BPPV Maneuver Guider")
st.image("diagnosis.gif", caption="Diagnosis", use_container_width=True)
st.image("bppv_treatment.gif", caption="Steps in BPPV", use_container_width=True)

webrtc_streamer(key="pose", video_processor_factory=PoseAnalyzer)
