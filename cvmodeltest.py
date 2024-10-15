import cv2
import mediapipe as mp
import numpy as np
import joblib
from sklearn.metrics import accuracy_score


exercises = ["pushup", "plank", "star_jumps", "squats"]
select = int(input("Train an exercise (0: pushup, 1: plank, 2: star_jumps, 3: squats): "))
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

model = joblib.load(f'C:/path/to/file/{exercises[select]}_model.joblib')

# Initialize video capture
# cap = cv2.VideoCapture(f"C:/path/to/file/{exercises[select]}_2.mp4")  # Use 0 for webcam, or provide a video file path
cap = cv2.VideoCapture(0)
# Get frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
time_per_frame = 1 / fps  # Time for each frame in seconds

def process_landmarks(results):
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        row = []
        for landmark in landmarks:
            row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(row).reshape(1, -1)
    return None

# Initialize variables for accuracy and rep counting
true_labels = []  # You'll need to provide these based on your ground truth data
predicted_labels = []
rep_count = 0
previous_phase = None
plank_start_frame = None
plank_total_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    # Draw landmarks on the frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )

        # Process landmarks and predict
        landmark_data = process_landmarks(results)
        if landmark_data is not None:
            prediction = model.predict(landmark_data)
            predicted_labels.append(prediction[0])

            if exercises[select] == "plank":
                if prediction[0] == 'up' and plank_start_frame is None:
                    # Start plank
                    plank_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                elif prediction[0] != 'up' and plank_start_frame is not None:
                    # End plank and calculate time spent in the plank
                    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    plank_duration = (current_frame - plank_start_frame) * time_per_frame
                    plank_total_time += plank_duration
                    plank_start_frame = None  # Reset for the next plank
            
            # Count reps (assuming 'up' phase completes a rep)
            if exercises[select] != "plank" and prediction[0] == 'up' and previous_phase == 'down':
                rep_count += 1
                
            previous_phase = prediction[0]
            
            # Display prediction and rep count on frame
            cv2.putText(frame, f"Phase: {prediction[0]}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Reps: {rep_count}" if exercises[select] != "plank" else f"Plank Time: {plank_total_time:.2f}s", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Exercise Phase Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate accuracy
if true_labels:  # Only calculate if true labels are provided
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.2f}")

print(f"Total reps: {rep_count}")

# Release resources
cap.release()
cv2.destroyAllWindows()