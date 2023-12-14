import os
import json
import random
import numpy as np
from tqdm import tqdm


NUM_SAMPLE_POINTS = 2000

#jsonから読み込む作業
object_dir_path = '../dataset_pointnet/pointcloud'
object_names = os.listdir(object_dir_path)
point_clouds = []
save_dir = '../dataset_pointnet_normalized/pointcloud'


for object_name in tqdm(object_names):
    object_path = os.path.join(object_dir_path,object_name)
    ids = os.listdir(object_path)
    
    #もしsave_dirの先が作られいなければ作成しておく
    if not os.path.exists(os.path.join(save_dir,object_name)):
        os.makedirs(os.path.join(save_dir,object_name))
    
    for id in ids: #id ->00001.json的な
        id_path = os.path.join(object_path,id)
        
        with open(id_path,'r') as file:
            loaded_data = json.load(file)
        
        current_point_cloud = loaded_data["coords"] #点群の2次元配列？
        num_points = len(current_point_cloud)
        
        #ランダムサンプリング
        sampled_indices = random.sample(list(range(num_points)),NUM_SAMPLE_POINTS)
        sampled_point_cloud = np.array([current_point_cloud[i] for i in sampled_indices])
        
        #正規化
        norm_point_cloud = sampled_point_cloud - np.mean(sampled_point_cloud,axis=0)
        norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud,axis=1))
        # point_clouds[index] = norm_point_cloud #配列に入れてるだけ
        #ラベルは必要ない
        
        #保存してすぐに取り出せる形に戻す
        data_to_save = {'coords':norm_point_cloud}    
        file_name = os.path.splitext(os.path.basename(id))[0]
        json_path = os.path.join(save_dir,object_name,id)
        
        with open(norm_point_cloud,'w') as file:
            json.dump(data_to_save,file)
          
          
            
            
        
        
        


# for index in tqdm(range(len(point_clouds))):
#   current_point_cloud = point_clouds[index]
#   current_label_cloud = point_cloud_labels[index]
#   current_labels = all_labels[index]
#   num_points = len(current_point_cloud)
#   # ランダムサンプリング
#   sampled_indices = random.sample(list(range(num_points)), NUM_SAMPLE_POINTS)
#   sampled_point_cloud = np.array([current_point_cloud[i] for i in sampled_indices])
#   sampled_label_cloud = np.array([current_label_cloud[i] for i in sampled_indices])
#   sampled_labels = np.array([current_labels[i] for i in sampled_indices])

#   # 正規化 
#   norm_point_cloud = sampled_point_cloud - np.mean(sampled_point_cloud, axis=0)
#   norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud, axis=1))
#   point_clouds[index] = norm_point_cloud
#   point_cloud_labels[index] = sampled_label_cloud
#   all_labels[index] = sampled_labels