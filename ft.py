import cv2
import numpy as np
import mediapipe as mp
import time
import tkinter as tk
from threading import Thread

# Setup MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Angle calculation
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

# Helper to get landmark points
def get_point(landmarks, name):
    lm = landmarks[mp_pose.PoseLandmark[name].value]
    return [lm.x, lm.y]

# Main detection logic
def detect_exercise_and_count(landmarks):
    global pushup_count, squat_count, jack_count, crunch_count, plank_duration
    global pushup_stage, squat_stage, jack_stage, crunch_stage, plank_start_time

    feedback = ""

    # Points
    ls, le, lw = get_point(landmarks, 'LEFT_SHOULDER'), get_point(landmarks, 'LEFT_ELBOW'), get_point(landmarks, 'LEFT_WRIST')
    lh, lk, la = get_point(landmarks, 'LEFT_HIP'), get_point(landmarks, 'LEFT_KNEE'), get_point(landmarks, 'LEFT_ANKLE')

    # Angles
    pushup_angle = calculate_angle(ls, le, lw)
    squat_angle = calculate_angle(lh, lk, la)
    plank_angle = calculate_angle(ls, lh, la)
    crunch_angle = calculate_angle(le, ls, lh)

    # Push-ups
    if pushup_angle > 160:
        pushup_stage = "up"
    if pushup_angle < 90 and pushup_stage == "up":
        pushup_stage = "down"
        pushup_count += 1
        feedback = "Good Push-Up!"

    # Squats
    if squat_angle > 160:
        squat_stage = "up"
    if squat_angle < 90 and squat_stage == "up":
        squat_stage = "down"
        squat_count += 1
        feedback = "Nice Squat!"

    # Jumping Jacks
    hand_dist = abs(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y - landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y)
    foot_dist = abs(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x - landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x)
    if hand_dist < 0.2 and foot_dist > 0.4:
        jack_stage = "open"
    elif hand_dist > 0.3 and foot_dist < 0.3 and jack_stage == "open":
        jack_stage = "close"
        jack_count += 1
        feedback = "Great Jumping Jack!"

    # Plank
    if 160 < plank_angle < 180:
        if plank_start_time == 0:
            plank_start_time = time.time()
        plank_duration = int(time.time() - plank_start_time)
        feedback = "Hold the plank!"
    else:
        plank_start_time = 0
        plank_duration = 0

    # Crunches
    if crunch_angle < 60:
        crunch_stage = "down"
    if crunch_angle > 100 and crunch_stage == "down":
        crunch_stage = "up"
        crunch_count += 1
        feedback = "Great Crunch!"

    return feedback

# Video and pose logic
def run_pose_estimation():
    global pushup_label, squat_label, jack_label, plank_label, crunch_label, feedback_label
    global pushup_count, squat_count, jack_count, crunch_count, plank_duration
    global pushup_stage, squat_stage, jack_stage, crunch_stage, plank_start_time

    pushup_count = squat_count = jack_count = crunch_count = 0
    pushup_stage = squat_stage = jack_stage = crunch_stage = None
    plank_start_time = plank_duration = 0

    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                feedback = detect_exercise_and_count(landmarks)

                pushup_label.config(text=f"Push-Ups: {pushup_count}")
                squat_label.config(text=f"Squats: {squat_count}")
                jack_label.config(text=f"Jumping Jacks: {jack_count}")
                plank_label.config(text=f"Plank Time: {plank_duration}s")
                crunch_label.config(text=f"Crunches: {crunch_count}")
                feedback_label.config(text=f"Feedback: {feedback}")

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            except:
                pass

            cv2.imshow('Virtual Fitness Trainer', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# GUI
root = tk.Tk()
root.title("Virtual Fitness Trainer")
root.geometry("400x300")
root.configure(bg="#0f172a")

tk.Label(root, text="AI Fitness Trainer", font=("Segoe UI", 18), fg="white", bg="#0f172a").pack(pady=10)
pushup_label = tk.Label(root, text="Push-Ups: 0", font=("Segoe UI", 14), fg="cyan", bg="#0f172a")
pushup_label.pack()
squat_label = tk.Label(root, text="Squats: 0", font=("Segoe UI", 14), fg="yellow", bg="#0f172a")
squat_label.pack()
jack_label = tk.Label(root, text="Jumping Jacks: 0", font=("Segoe UI", 14), fg="orange", bg="#0f172a")
jack_label.pack()
plank_label = tk.Label(root, text="Plank Time: 0s", font=("Segoe UI", 14), fg="lightgreen", bg="#0f172a")
plank_label.pack()
crunch_label = tk.Label(root, text="Crunches: 0", font=("Segoe UI", 14), fg="magenta", bg="#0f172a")
crunch_label.pack()
feedback_label = tk.Label(root, text="Feedback: ", font=("Segoe UI", 12), fg="white", bg="#0f172a")
feedback_label.pack(pady=5)

tk.Button(root, text="Start Trainer", font=("Segoe UI", 12), bg="#38bdf8", fg="white",
          command=lambda: Thread(target=run_pose_estimation, daemon=True).start()).pack(pady=10)

root.mainloop()
