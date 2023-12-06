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
object_dir_path ='../dataset_pointnet/pointcloud'
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

        pc = PointCloud(
            coords=np.array(loaded_data['coords']),
            channels = {}
            # channels={key:np.array(value) for key, value in loaded_data['channels'].items()}
        )          

        # Plot the point cloud as a sanity check.
        fig = plot_point_cloud(pc, grid_size=2)

        # Produce a mesh (with vertex colors)
        mesh = marching_cubes_mesh( #エラー
        pc=pc,
        model=model,
        batch_size=4096,
        grid_size=32, # increase to 128 for resolution used in evals
        progress=True,
        )

        #保存
        save_top_directory = '../dataset_pointnet/polygon' 
        save_middle_directory = os.path.join(save_top_directory,object_name)
        if not os.path.exists(save_middle_directory):
            os.makedirs(save_middle_directory)
            
        save_path = os.path.join(save_top_directory,object_name,f"{only_id}.ply")
        # Write the mesh to a PLY file to import into some other program.
        with open(save_path, 'wb') as f:
            mesh.write_ply(f)

