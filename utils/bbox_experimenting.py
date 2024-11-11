import os
import cv2
import pandas as pd
import numpy as np

def display_bbox_on_videos(video_dir, dataframe):
    """
    Reads videos one by one, matches each video with its corresponding row in the DataFrame,
    reads the 'bbox' column with normalized coordinates, and displays the bounding box on all frames.

    Args:
    video_dir (str): Path to the directory containing the videos.
    dataframe (pd.DataFrame): DataFrame containing video information, including 'bbox' column.

    Returns:
    None
    """
    # Get a list of all video files in the directory
    video_files = [f for f in os.listdir(video_dir)]

    if not video_files:
        print("No video files found in the specified directory.")
        return

    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_dir, video_file)

        # Find the corresponding row in the DataFrame
        # Assuming there is a column 'video_name' or 'youtube_id' to match
        # Modify this part based on your DataFrame's structure
        matching_rows = dataframe[dataframe['youtube_id'] == video_name]

        if matching_rows.empty:
            print(f"No matching data found for video {video_file}")
            continue

        # Assuming the DataFrame has one row per video
        row = matching_rows.iloc[0]

        # Extract normalized bbox coordinates from the 'bbox' column
        # The 'bbox' column is expected to be a string like '[x_min y_min x_max y_max]'
        bbox_norm = row['bbox']
        # bbox_str = bbox_str.strip('[]').split()
        # bbox_norm = [float(coord) for coord in bbox_str]  # [x_min_norm, y_min_norm, x_max_norm, y_max_norm]

        # Load the video
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            continue

        print(f"Displaying bounding box on video: {video_file}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            height, width, _ = frame.shape

            # Convert normalized bbox coordinates to pixel values
            x_min = int(bbox_norm[0] * width)
            y_min = int(bbox_norm[1] * height)
            x_max = int(bbox_norm[2] * width)
            y_max = int(bbox_norm[3] * height)

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

            # Display the frame
            cv2.imshow('Video with Bounding Box', frame)

            # Press 'q' to exit the visualization
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    print("Completed displaying bounding boxes on all videos.")

# Example usage
video_directory = 'dataset/data/orig_videos'  # Replace with your video directory path
dataframe_path = 'dataset/data/golfDB.pkl'     # Replace with the path to your DataFrame CSV file

# Load the DataFrame
df = pd.read_pickle(dataframe_path)

# Assuming the DataFrame has a 'video_name' column to match videos
# If your DataFrame uses 'youtube_id' or another identifier, adjust accordingly

display_bbox_on_videos(video_directory, df)