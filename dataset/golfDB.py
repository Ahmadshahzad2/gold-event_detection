import pandas as pd
import os
import yt_dlp
import cv2 
import mediapipe as mp
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

# from ..utils.general_utils import load_json
import json

def load_json(file_path):
    """
    Load and return the contents of a JSON file.

    Parameters:
    - file_path (str): The path to the JSON file.

    Returns:
    - dict: The contents of the JSON file as a Python dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print("The file was not found. Please check the file path.")
        return None
    except json.JSONDecodeError:
        print("The file is not a valid JSON. Please check the JSON syntax.")
        return None


class GolfDB():
    def __init__(self,data_path='dataset/data/golfDB.pkl',video_dir='dataset/data/orig_videos', pose_dir='dataset/data/orig_pose',mp_model='weights/pose_landmarker.task',data_point_path='dataset/data/data_points',shot_seg_len=30,stride=30 ):
        self.df=pd.read_pickle(data_path)
        self.video_dir=video_dir
        self.pose_dir=pose_dir
        self.mp_model_path=mp_model
        self.df_back=self.add_background()
        self.shot_seg_len=shot_seg_len
        self.data_point_path=data_point_path
        self.stride=stride



    def download_youtube_videos(self, youtube_ids):
        """
        Downloads a list of YouTube videos using their IDs.

        Args:
        youtube_ids (list): List of YouTube video IDs to download.
        output_path (str): Directory where videos will be saved.

        Returns:
        None
        """
        links = [f"https://www.youtube.com/watch?v={youtube_id}" for youtube_id in youtube_ids]

        for link in links:
            try:
                # Set options for yt-dlp
                ydl_opts = {
                    'outtmpl': f'{ self.video_dir}/%(id)s.%(ext)s',  # Custom filename with video ID
                    'format': 'bestvideo+bestaudio/best',        # Get best video and audio available
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    print(f"Downloading from {link}...")
                    ydl.download([link])
                    print("Download complete!")

            except Exception as e:
                print(f"Error downloading {link}: {e}")

        print('Task Completed!')

    def get_videos_data(self):
        yt_ids=self.df['youtube_id']
        self.download_youtube_videos(yt_ids,self.video_dir)

    def _initialize_pose_landmarker(self):
        """
        Initializes the MediaPipe Pose Landmarker with video mode.
        """
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a pose landmarker instance with the video mode
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='weights/pose_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO
        )

        pose_landmarker = PoseLandmarker.create_from_options(options)
        return pose_landmarker

    def get_video_pose_landmark(self):
        """
        Iterates over all videos in a directory, processes each video,
        and saves the pose landmarks into JSON files.

        Args:
        video_directory (str): Path to the directory containing videos.
        output_directory (str): Path to the directory where JSON files will be saved.
        """


        # Get a list of all video files in the directory
        video_files = [f for f in os.listdir(self.video_dir)]  # Add more extensions if needed

        for i,video_file in enumerate(video_files):
            self.pose_landmarker = self._initialize_pose_landmarker()

            video_path = os.path.join(self.video_dir, video_file)
            output_file = os.path.splitext(video_file)[0] + '.json'
            output_path = os.path.join(self.pose_dir, output_file)

            print(f'processing {i}/{len(video_files)}')


            if os.path.exists(output_path):
                print('Already processed')
                continue

            print(f"Processing video: {video_file}")

            # Get the pose data dictionary for the video
            pose_data = self.get_video_dict(video_path)

            # Save the pose data to a JSON file
            with open(output_path, 'w') as f:
                json.dump(pose_data, f,indent=4)

            print(f"Pose data saved to {output_file}")

    def get_video_dict(self, video_path):
        """
        Processes the video, extracts pose landmarks for each frame,
        and returns a dictionary with frame indices as keys and landmarks as values.

        Args:
        video_path (str): Path to the video file.

        Returns:
        dict: A dictionary containing pose landmarks for each frame.
        """
        pose_data = {}
        cap = cv2.VideoCapture(video_path)

        # Get the frames per second (fps) of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None or fps != fps:  # fps != fps checks for NaN
            fps = 30  # Default to 30 fps if fps is invalid
            print(f"Warning: fps is invalid for {video_path}. Defaulting to {fps} fps.")
        frame_index = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            print(video_path, frame_index)

            # Convert the image to RGB as required by MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Calculate the timestamp in milliseconds
            timestamp_ms = (frame_index / fps) * 1000

            # Process the frame and get pose landmarks
            detection_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms=int(timestamp_ms))
            landmarks_list = []

            # Extract landmarks if detected
            if detection_result.pose_landmarks:
                for landmark in detection_result.pose_landmarks[0]:
                    landmarks_list.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })

                # Use frame index as the key
            pose_data[frame_index] = landmarks_list

            frame_index += 1

        cap.release()
        return pose_data

    def add_background(self):
        filtered_df=self.df[self.df['slow']!=1]
        filtered_df['Label']='shot_phase'


        # filtered_df['events_list'] = filtered_df['events'].apply(
        # lambda x: list(map(int, x.strip('[]').split())) if isinstance(x, str) else []
        # )

        new_rows = []
        start_background_frame_len=90
        end_background_frame_len=30


        for index, row in filtered_df.iterrows():
            # Parse the 'events' column to get the list of frames
            # Convert the string representation of the list to an actual list of integers
            events_list = row['events']
            #  =list(map(int, re.sub(r'[^\d\s]', '', events_str).split()))

            

            # events_list = row['events_list']
            # print(events_list)

            # Get the start and end frames of the original event
            start_frame = events_list[1]
            end_frame = events_list[-2]

            # Create the first new row (background before the event)
            new_row_before = row.copy()
            new_row_before['Label'] = 'Background'

            # Calculate the frames for the background event before the start frame
            background_start_before = max(0, start_frame - start_background_frame_len)  # Ensure non-negative frame index
            background_end_before = start_frame

            # Update the 'events' column
            new_row_before['events'] = f'[{background_start_before} {background_end_before}]'

            # Create the second new row (background after the event)
            new_row_after = row.copy()
            new_row_after['Label'] = 'Background'

            # Calculate the frames for the background event after the end frame
            background_start_after = end_frame
            background_end_after = end_frame + end_background_frame_len  # Adjust if total frames are known

            # Update the 'events' column
            new_row_after['events'] = f'[{background_start_after} {background_end_after}]'

            # Append the new rows to the list
            new_rows.append(new_row_before)
            new_rows.append(new_row_after)

            # Convert the list of new rows to a DataFrame
            new_rows_df = pd.DataFrame(new_rows)

            # Append the new rows to the original filtered DataFrame
            final_df = pd.concat([filtered_df, new_rows_df], ignore_index=True)

            # Display the final DataFrame
        return final_df

    def get_shot_data_points(self,shot_class_mapping={'Background':0,'shot_phase':1 },set_fps=30):

        data_points_spec=f'{self.data_point_path}/seg_len{self.shot_seg_len}_stride{self.stride}_fps{set_fps}'

        os.makedirs(data_points_spec,exist_ok=True)

        for index, row  in self.df_back.iterrows():

            count=1
            label=row['Label']
            youtube_id=row['youtube_id']
            events=row['events']

            json_path=os.path.join(self.pose_dir,youtube_id+'.json')
            pose_dict=load_json(json_path)

            fps=pose_dict['video_meta_data']['fps']
            width= pose_dict['video_meta_data']['width']
            height=pose_dict['video_meta_data']['height']

            if round(fps)>set_fps:
                continue

            if label=='Background':
    
                start,end=events

            else:
                start,end=events[1],events[-1]

            data_sequence=[]

            for i in range(start+self.shot_seg_len,end+1,self.stride):

                for j in range(i-self.shot_seg_len,i+1):

                    frame_data=pose_dict[str(j)]
                    if frame_data:
                        frame_cords=[[keypoint['x']*width, keypoint['y']*height] for keypoint in frame_data]
                        data_sequence.append(frame_cords)

                        if len(data_sequence)==self.shot_seg_len:
                            save_path = os.path.join(self.data_point_path, f'{youtube_id}_{count}.pkl')
                            with open(save_path, 'wb') as f:
                                pickle.dump((data_sequence,shot_class_mapping[label]), f)
                            data_sequence=[]
                            count+=1

    def update_video_metadata(self):
        """
        Loads each video and its corresponding JSON file, extracts resolution and FPS, 
        adds this information to the JSON under 'video_meta_data', and saves the updated JSON.

        Args:
        video_directory (str): Path to the directory containing videos.
        output_directory (str): Path to the directory where JSON files are saved.
        """
        
        # Get a list of all video files in the directory
        video_files = [f for f in os.listdir(self.video_dir)]

        for i, video_file in enumerate(video_files):
            video_path = os.path.join(self.video_dir, video_file)
            output_file = os.path.splitext(video_file)[0] + '.json'
            output_path = os.path.join(self.pose_dir, output_file)

            # Check if the JSON file exists
            if not os.path.exists(output_path):
                print(f'JSON file for {video_file} does not exist, skipping.')
                continue

            print(f"Updating metadata for video: {video_file}")

            # Load the JSON file
            pose_data = load_json(output_path)

            # Extract video metadata using OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open video: {video_file}")
                continue

            # Get resolution and FPS
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # Add metadata to the JSON
            pose_data['video_meta_data'] = {
                "width": width,
                'height': height,
                "fps": fps
            }

            # Save the updated JSON file
            with open(output_path, 'w') as f:
                json.dump(pose_data, f, indent=4)

            print(f"Updated metadata for {output_file}")


class golfDB_Dataset(Dataset):
    def __init__(self, pose_dir,class_mapping, transform=None):
        self.folder_path = pose_dir
        self.transform = transform
        self.class_mapping=class_mapping

        self.file_list = [f for f in os.listdir(pose_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_list[idx])
        npy_data = np.load(file_path)  # Load the npy file
        
        data = npy_data[0]  #  data is stored in the first index
        label = self.class_mapping[npy_data[1]]  # label is stored in the second index

        if self.transform:
            data = self.transform(data)

        return data, label





if __name__ == '__main__':


    # from dataset import GolfDB
    import pandas as pd

    # df_dir='/Users/ahmadshahzad/Documents/Office_work/Golf-event_detection_revamped/dataset/data/golfDB.pkl'

    # df=pd.read_pickle(df_dir)
    # golf_data=GolfDB(df_dir)

    # golf_data.get_video_pose_landmark()

    # golf_data.get_data_points()

    dataset=golfDB_Dataset(pose_dir='dataset/data/data_points')







        
        