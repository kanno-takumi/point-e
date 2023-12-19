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

file_name ='pointcloud_data/generated/normalized/S__34930732.json'
with open(file_name, 'r') as file:
    loaded_data = json.load(file)

# 色情報がある場合のプロット
if 'channels' in loaded_data and loaded_data['channels']:
    loaded_pc = PointCloud(
        coords=np.array(loaded_data['coords']),
        channels={key: np.array(value) for key, value in loaded_data['channels'].items()}
    )      
    fig_with_color = plot_point_cloud(loaded_pc, color=True, grid_size=1, fixed_bounds=((-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)))
    
# 色なし（単色）の場合のプロット
else:
    loaded_pc = PointCloud(
        coords=np.array(loaded_data['coords']),
        channels=None
    )
    fig_without_color = plot_point_cloud(loaded_pc, color=False, grid_size=1, fixed_bounds=((-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)))

plt.show()


