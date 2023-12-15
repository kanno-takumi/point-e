import os
import json
import random
import numpy as np
from tqdm import tqdm

NUM_SAMPLE_POINTS = 2000

with open(path,'r') as file:
    loaded_data = json.load(file)
    
current_point_cloud = loaded_data["coords"] #点群の2次元配列？
num_points = len(current_point_cloud)


#サンプリング
sampled_indices = random.sample(list(range(num_points)),NUM_SAMPLE_POINTS)
sampled_point_cloud = np.array([current_point_cloud[i] for i in sampled_indices])


#正規化
norm_point_cloud = sampled_point_cloud - np.mean(sampled_point_cloud,axis=0)
norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud,axis=1))

norm_point_cloud_list = norm_point_cloud.tolist() #リストへ変換
data_to_save = {'coords':norm_point_cloud_list}    
json_path = path
        
with open(json_path,'w') as file:
    json.dump(data_to_save,file)