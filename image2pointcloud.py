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
import json

# モデルの準備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base40M' # use base300M or base1B for better results
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('downloading base checkpoint...')
base_model.load_state_dict(load_checkpoint(base_name, device))

print('downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))

# サンプラーの準備
sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 3.0],
)

# 画像の準備
img = Image.open('input_picture/cup.jpg')

# 推論の実行
samples = None
for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
    samples = x
    
# ポイントクラウドの表示
pc = sampler.output_to_point_clouds(samples)[0] #pc=pointcloud

#データを保存するファイル名
file_name = 'pointcloud_data/cup.json'

#型がPointCloudデータになっているため、辞書型に直す
data_to_save = {
    'coords':pc.coords.tolist(),
    'channels': {key:value.tolist() for key,value in pc.channels.items()}
}

with open(file_name,'w') as file:
    json.dump(data_to_save,file)



#ここの引数でgrid_sizeを決めている 3にすると3*3=9表示される
fig = plot_point_cloud(pc, grid_size=1, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))
print(type(fig))
#f2 = open(('fig.txt','a')) 
#f2.write(str(fig))
#f2.close()

#plt.show()
