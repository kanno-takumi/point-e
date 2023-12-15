#今はフォルダ名がlabel - jsonたち - pointcloud
#pointcloud,labelで保存したい。(ランダムである必要はない)

from tabnanny import filename_only
from PIL import Image
import torch 
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud
import json
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating SDF model...')
name = 'sdf'
model = model_from_config(MODEL_CONFIGS[name], device)
model.eval()

print('loading SDF model...')
model.load_state_dict(load_checkpoint(name, device))

# Load a point cloud we want to convert into a mesh.
#pc = PointCloud.load('example_data/pc_corgi.npz')

# 文字列からデータ構造に変換
object_dir_path ='../dataset_pointnet_normalized/pointcloud'
object_names = os.listdir(object_dir_path)

for object_name in tqdm(object_names): #Object->Tableみたいなこと
    object_path =os.path.join(object_dir_path,object_name) #/./././Tableみたいなこと
    ids = os.listdir(object_path)
    
    for id in ids:
        id_path = os.path.join(object_path,id)
        only_id = os.path.splitext(id)[0] #拡張子なし
        
        #読み込み
        with open(id_path,'r') as file:
            loaded_data = json.load(file)

        dataset = {
            'coords' : np.array(loaded_data['coords']), 
            'label' : object_name
            }  

        #保存
        save_dir = '../dataset_pointnet_normalized/pcandlabel' 
            
        save_path = os.path.join(save_dir,id)
        # Write the mesh to a PLY file to import into some other program.
        with open(save_path, 'w') as f: #plyはバイナリファイルだからwbにしなければいけない。
            json.dump(dataset,f)



