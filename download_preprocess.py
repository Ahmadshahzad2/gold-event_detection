# %%
import pandas as pd
pd.set_option('display.max_columns', None)  # Show all columns


df_dir='dataset/data/golfDB.pkl'

df=pd.read_pickle('dataset/data/golfDB.pkl')
df.head()
# df.to_csv('dataset/data/golfDB.csv')

# %%
df.head()

# %%
def last_two_diff(events):
    if len(events) >= 2:
        return events[-1] - events[-2]
    else:
        return None  # or any other default value like 0 if lists might have less than two elements

seg_len = df['events'].apply(last_two_diff)

# %%
# import matplotlib.pyplot as plt
# # Plotting the distribution of the differences
# plt.figure(figsize=(10, 6))
# plt.hist(seg_len, bins=30, alpha=0.75, edgecolor='black')
# plt.title('Distribution of Differences between Last Two Events')
# plt.xlabel('Difference')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

# %%
# import os
# import json
# from utils import load_json
# def get_points(self,seg_len=30):
#     point_path='dataset/data/orig_pose'

#     for path in os.listdirs(point_path):
#         json_pth=os.path.join(point_path,path)
#         keypoint=load_json(json_pth)




    



# %%
from dataset import GolfDB
golf_data=GolfDB(df_dir,method='mediapipe')

# %%
# golf_data.get_videos_data('dataset/data/orig_videos')    

# %%
golf_data.get_video_pose_landmark() ## Done

# golf_data.update_video_metadata()

# %%
# golf_data.get_shot_data_points()


