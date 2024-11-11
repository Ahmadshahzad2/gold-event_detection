import os
import cv2
import json
import numpy as np
import pandas as pd

def visualize_landmarks_with_cv2_points_only(video_dir, json_dir, df_back, output_dir):
    """
    Iterates through each video file and its corresponding JSON file,
    overlays the pose landmarks on frames within the specified range,
    and saves the processed frames as a new video file.

    Args:
    video_dir (str): Path to the directory containing video files.
    json_dir (str): Path to the directory containing JSON files with landmarks.
    df_back (pd.DataFrame): DataFrame containing video information with min and max frame indices.
    output_dir (str): Path to the directory where processed videos will be saved.

    Returns:
    None
    """
    
    # Define the function to find min and max frames based on the DataFrame
    def find_max_min(video_id):
        filtered_df = df_back[df_back['youtube_id'] == video_id]
        if filtered_df.empty:
            return None, None
        all_lists = filtered_df['events']
        flattened_values = [item for sublist in all_lists for item in sublist]
        min_value = min(flattened_values)
        max_value = max(flattened_values)
        return min_value, max_value

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        print("No video files found in the specified directory.")
        return

    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_dir, video_file)

        # Find the corresponding JSON file
        json_file = video_name + '.json'
        json_path = os.path.join(json_dir, json_file)

        if not os.path.exists(json_path):
            print(f"JSON file {json_file} not found for video {video_file}. Skipping this video.")
            continue

        # Get min and max frame indices for this video
        min_frame, max_frame = find_max_min(video_name)
        if min_frame is None or max_frame is None:
            print(f"No frame range found for video {video_file}. Skipping.")
            continue

        # Load the landmarks from the JSON file
        try:
            with open(json_path, 'r') as f:
                pose_data = json.load(f)
        except Exception as e:
            print(f"Could not load JSON file for {video_file}. Skipping. Error: {e}")
            continue

        # Create a VideoCapture object
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            continue

        # Check frame rate and dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        if frame_width == 0 or frame_height == 0:
            print(f"Could not retrieve frame dimensions for {video_file}. Skipping.")
            cap.release()
            continue

        if frame_rate == 0 or frame_rate is None or np.isnan(frame_rate):
            print(f"Could not retrieve frame rate for {video_file}. Setting default frame rate to 30 FPS.")
            frame_rate = 30  # Default frame rate

        # Set up VideoWriter for output
        output_video_path = os.path.join(output_dir, f"{video_name}_with_landmarks.mp4")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width, frame_height))

        frame_index = 0

        print(f"Processing video: {video_file}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process frames within the min and max frame range
            if frame_index < min_frame or frame_index > max_frame:
                frame_index += 1
                continue

            # Get landmarks for the current frame
            landmarks_list = pose_data.get(str(frame_index), None)

            if landmarks_list:
                height, width, _ = frame.shape

                # Convert normalized landmarks to pixel coordinates
                for lm in landmarks_list:
                    x_px = int(lm['x'] * width)
                    y_px = int(lm['y'] * height)

                    # Draw the landmark point
                    cv2.circle(frame, (x_px, y_px), radius=3, color=(0, 255, 0), thickness=-1)

            # Write the frame to the output video
            out.write(frame)

            frame_index += 1

        cap.release()
        out.release()
        print(f"Saved processed video to {output_video_path}")

    print("Processing completed.")

# Example usage
video_directory = 'dataset/data/orig_videos'  # Replace with your video directory path
json_directory = 'dataset/data/orig_pose/mediapipe/full'  # Replace with your JSON directory path
output_directory = 'dataset/data/pipeline_recheck_vids'  # Replace with your output directory path
df_back = pd.read_pickle('dataset/data/golfDB_background_added.pkl')

visualize_landmarks_with_cv2_points_only(video_directory, json_directory, df_back, output_directory)
