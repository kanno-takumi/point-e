import open3d as o3d
import numpy as np

if __name__ == "__main__":
    
    # Loading point cloud
    print("Loading point cloud")
    #ptCloud = o3d.io.read_point_cloud("dataset/trimesh_primitives/train/0/box_0000.ply")
    ptCloud = o3d.io.read_point_cloud("mesh_data/cup.ply")

    # confirmation
    print(ptCloud)
    print(np.asarray(ptCloud.points))
    
    # Visualization in window
    o3d.visualization.draw_geometries([ptCloud])
    
    # Saving point cloud
    #o3d.io.write_point_cloud("output.ply", ptCloud)　#すでに保存してあるため必要ない