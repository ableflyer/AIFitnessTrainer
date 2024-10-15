import cv2
import mediapipe as mp
import csv
import os

# Define the exercise types
exercises = ["pushup", "plank", "star_jumps", "squats"]
select = int(input("Select an exercise (0: pushup, 1: plank, 2: star_jumps, 3: squats): "))

# Initialize MediaPipe Pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize OpenCV for video capture
cap = cv2.VideoCapture(f"C:/path/to/file/{exercises[select]}_2.mp4")

# Filepath of the CSV file
csv_file = f'{exercises[select]}_coordinates.csv'

# Create header for CSV
header = ['phase'] + [f'landmark_{landmark}_{axis}' for landmark in range(33) for axis in ['x', 'y', 'z', 'visibility']]

# Check if the CSV file exists, if not, create it and add the header
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode='a', newline='') as file:
    csv_writer = csv.writer(file)
    
    # If the file doesn't exist or is empty, add the header
    if not file_exists or os.stat(csv_file).st_size == 0:
        csv_writer.writerow(header)
    
    frame_count = 0
    current_phase = 'unknown'  # Can be 'up' or 'down'
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to capture.")
            break
        
        # Convert the frame to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Pose detection
        results = pose.process(rgb_frame)
        
        # Draw landmarks on the frame if detected
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
        
        # Display the current phase on the frame
        cv2.putText(frame, f"Current Phase: {current_phase}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show the video
        cv2.imshow('MediaPipe Pose Tracking', frame)

        # Capture keypress to set phase ('u' for up, 'd' for down, 'q' to quit)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('u'):
            current_phase = 'up'
            landmarks = results.pose_landmarks.landmark
            row = ["up"]
            for landmark in landmarks:
                row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
            # Append the row to the CSV
            csv_writer.writerow(row)
        elif key == ord('d'):
            current_phase = 'down'
            landmarks = results.pose_landmarks.landmark
            row = ["down"]
            for landmark in landmarks:
                row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
            # Append the row to the CSV
            csv_writer.writerow(row)
        elif key == ord('q'):
            break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()