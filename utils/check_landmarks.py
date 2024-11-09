import os
import cv2
import json
import random
import mediapipe as mp
import numpy as np

def visualize_landmarks_with_cv2_points_only(video_dir, json_dir):
    """
    Randomly selects a video and its corresponding JSON file,
    overlays the pose landmarks on each frame using cv2 (without connections),
    and displays the frames.

    Args:
    video_dir (str): Path to the directory containing video files.
    json_dir (str): Path to the directory containing JSON files with landmarks.

    Returns:
    None
    """
    # Get a list of video files
    video_files = [f for f in os.listdir(video_dir)
                   if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        print("No video files found in the specified directory.")
        return

    # Randomly select a video
    selected_video = random.choice(video_files)
    video_name = os.path.splitext(selected_video)[0]
    video_path = os.path.join(video_dir, selected_video)

    # Find the corresponding JSON file
    json_file = video_name + '.json'
    json_path = os.path.join(json_dir, json_file)

    if not os.path.exists(json_path):
        print(f"JSON file {json_file} not found in the specified directory.")
        return

    # Load the landmarks from the JSON file
    with open(json_path, 'r') as f:
        pose_data = json.load(f)

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    frame_index = 0

    print(f"Visualizing landmarks for video: {selected_video}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get landmarks for the current frame
        landmarks_list = pose_data.get(str(frame_index), None)

        if landmarks_list:
            height, width, _ = frame.shape

            # Convert normalized landmarks to pixel coordinates
            for lm in landmarks_list:
                print(width,height)
                x_px = int(lm['x'] * width)
                y_px = int(lm['y'] * height)

                # Draw the landmark point
                cv2.circle(frame, (x_px, y_px), radius=3, color=(0, 255, 0), thickness=-1)

        # Display the frame
        cv2.imshow('Pose Landmarks', frame)

        # Press 'q' to exit the visualization
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_directory = 'dataset/data/orig_videos'  # Replace with your video directory path
json_directory = 'dataset/data/orig_pose'    # Replace with your JSON directory path

visualize_landmarks_with_cv2_points_only(video_directory, json_directory)
