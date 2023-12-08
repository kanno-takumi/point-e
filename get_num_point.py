import json
import os

# ユーザーからの入力でファイルパスを取得
# file_name = input("Enter the file name : ")
# ファイルパスの構築
# file_path = os.path.join('pointcloud_data',file_name )

file_path = input("Enter the path to the json file : ")
file_path = os.path.join('../dataset_pointnet/pointcloud/')

data = None
with open(file_path) as file:
    data = json.load(file)

coords_data = data["coords"]
coords_num = len(coords_data)
print("点群の数", coords_num)