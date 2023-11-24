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
#from point_e.util.point_cloud import PointCloud


# 文字列からデータ構造に変換
f = open('pointcloud_data.txt','r')
data_str = f.read()
data = ast.literal_eval(data_str)
coords = np.array(data['coords'])

# NumPy配列に変換
channels = {'R': np.array(data['channels']['R']),
            'G': np.array(data['channels']['G']),
            'B': np.array(data['channels']['B'])}


pc = PointCloud(coords=coords,channels=channels)
f.close()
fig = plot_point_cloud(pc, grid_size=3 , fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))



