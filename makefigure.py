# パッケージのインポート
import sys,os
sys.path.append(os.pardir) 
from PIL import Image
import torch
from tqdm.auto import tqdm
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
import matplotlib.pyplot as plt
import ast
import numpy
from typing import Dict,Optional,Tuple
from point_e.util.point_cloud import PointCloud
import json
import numpy as np



# 文字列からデータ構造に変換
file_name ='pointcloud_data.json'
with open(file_name,'r') as file:
    loaded_data = json.load(file)

loaded_pc = PointCloud(
    coords=np.array(loaded_data['coords']),
    channels={key:np.array(value) for key, value in loaded_data['channels'].items()}
)

fig = plot_point_cloud(loaded_pc, grid_size=1 , fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))

plt.show()



