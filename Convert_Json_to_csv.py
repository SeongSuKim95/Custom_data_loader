import pandas as pd
import torch
import numpy as np
device = ("cuda" if torch.cuda.is_available() else "cpu")

json_df = pd.read_json(r"/home/sungsu21/Project/Data_Loader/result_cam2.json")

cnt = 0

for frame_num in range(json_df['frame_id'].max()):
    object_num = len(json_df['objects'][frame_num])
    if object_num:
        for object_index in range(object_num):
             if json_df['objects'][frame_num][object_index]['class_id'] == 0: 
                cnt +=1

csv_df = pd.DataFrame(index = range(cnt), columns=["frame_number","id","xmin","ymin","width","height","visibility","class","confidence"])
print(csv_df)
index = 0
for frame_num in range(json_df['frame_id'].max()):
    object_num = len(json_df['objects'][frame_num])
    if object_num:
        for object_index in range(object_num):
             if json_df['objects'][frame_num][object_index]['class_id'] == 0: 
                
                width = round(json_df['objects'][frame_num][object_index]['relative_coordinates']["width"] * 1920)
                height =  round(json_df['objects'][frame_num][object_index]['relative_coordinates']["height"] * 1080)
                csv_df["frame_number"][index] = frame_num + 1
                csv_df["xmin"][index] = round(json_df['objects'][frame_num][object_index]['relative_coordinates']["center_x"] *1920 - width/2)
                csv_df["ymin"][index] = round(json_df['objects'][frame_num][object_index]['relative_coordinates']["center_y"] *1080 - height/2)
                csv_df["width"][index]= width
                csv_df["height"][index] = height
                csv_df["confidence"][index] = json_df['objects'][frame_num][object_index]['confidence']

                index += 1

csv_df['id'] = -1
csv_df['visibility'] = 1
csv_df['class'] = 1   

csv_df.to_csv(r'result_cam2.csv',index = False, header = True)


