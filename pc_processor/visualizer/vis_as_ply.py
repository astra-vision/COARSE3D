import os

import numpy as np
import yaml
from plyfile import PlyData, PlyElement

# data_config_path = '/mnt/cephfs/home/lirong/code/poincloudProcessor-P2PContrast/pc_processor/dataset/semantic_kitti/config/labels/semantic-kitti.yaml'
# data_config_path = '/mnt/cephfs/home/lirong/code/poincloudProcessor-P2PContrast/pc_processor/dataset/semantic_poss/semantic-poss.yaml'
data_config_path = "/mnt/cephfs/home/lirong/code/poincloudProcessor-P2PContrast/pc_processor/dataset/nuScenes/nuscenes.yaml"
data_config = yaml.safe_load(open(data_config_path, "r"))


def save_ply(pcd, label=None, file_name="pcd", dir=None):
    xyz = pcd[:, :3]

    try:
        ref = pcd[:, 3]
    except:
        pass

    # ground_mask_file = pcd_file.replace('velodyne', 'ground_mask').replace('.bin', '.npy')
    # ground_mask = np.load(ground_mask_file).astype(np.int8)

    # labels[labels < 0] = 0  # 忽略-100
    # labels[ground_mask > 0] = 0

    # map 20 -> 260
    learning_map_inv = np.zeros((20,)).astype(np.int32)
    for key in data_config["learning_map_inv"].keys():
        # print(key, label_map[key])
        learning_map_inv[key] = data_config["learning_map_inv"][key]

    color = np.ones(xyz.shape)
    if not label is None:
        if label.max() < 20:
            label = learning_map_inv[label]
        assert pcd.shape[0] == label.shape[0]

        color_map = np.zeros((260, 3)).astype(np.int32)
        for key in data_config["color_map"].keys():
            # print(key, data_config['color_map'][key])
            color_map[key] = data_config["color_map"][key]
        for cls in np.unique(label):
            color[label == cls, :] = color_map[cls]
    else:
        import cv2

        ref = pcd[:, 3]
        color = cv2.applyColorMap((ref * 255).astype(np.uint8), cv2.COLORMAP_JET)
        depth = np.linalg.norm(xyz[:, :3], 2, axis=1)
        depth = (depth / depth.max()) * 255
        color = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)

    color = color.reshape(-1, 3)

    prop = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("blue", "u1"),
        ("green", "u1"),
        ("red", "u1"),
    ]

    vertex_all = np.empty(len(xyz), dtype=prop)  # 注意，np.empyty会赋予空值

    for i in range(0, 3):  # 画出整个scene中的所有点
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):  # 对不同的partition上不同的颜色
        vertex_all[prop[i + 3][0]] = color[:, i]

    # save file
    ply = PlyData([PlyElement.describe(vertex_all, "vertex")], text=True)

    # os.makedirs('/mnt/cephfs/home/lirong/code/poincloudProcessor-P2PContrast/', exist_ok=True)

    if dir is None:
        save_file = "/mnt/cephfs/home/lirong/code/poincloudProcessor-P2PContrast/plot_results/{}.ply".format(
            file_name
        )
    else:
        os.makedirs(dir, exist_ok=True)
        save_file = os.path.join(dir, "{}.ply".format(file_name))

    print(">> write a ply at ", save_file)
    ply.write(save_file)


if __name__ == "__main__":
    POINT_CLOUD_RANGE = [0, -40, -3, 70.4, 40, 1]
    x_max = POINT_CLOUD_RANGE[-3]
    y_max = POINT_CLOUD_RANGE[-2]
    z_max = POINT_CLOUD_RANGE[-1]
    range_max = [70.4, 40, 1]
    range_min = [0, -40, -3]

    save_plot_dir = "/mnt/cephfs/home/lirong/code/poincloudProcessor-P2PContrast/plot_results/0802/nuscenes/"

    # semantic kitti
    # salsa_path = '/mnt/cephfs/dataset/pointclouds/p2p_visualization/semantic-kitti/salsanext0.01/debug-False_0802_bs-01_1_ep-150_Model-SalsaNext_id-/plot/'
    # our_path = '/mnt/cephfs/dataset/pointclouds/p2p_visualization/semantic-kitti/ours0.01/debug-False_0802_id-/plot/'

    ## semantic poss
    # salsa_path = '/mnt/cephfs/dataset/pointclouds/p2p_visualization/semantic-poss/salsanext0.01/debug-False_0802_bs-01_1_ep-150_Model-SalsaNext_id-/plot/'
    # our_path = '/mnt/cephfs/dataset/pointclouds/p2p_visualization/semantic-poss/ours0.01/debug-False_0802_id-semantic-poss/plot/'

    # nuscenes
    salsa_path = "/mnt/cephfs/dataset/pointclouds/p2p_visualization/nuscenes/salsanext0.01/debug-False_0802_bs-01_1_ep-150_Model-SalsaNext_id-val/plot/"
    our_path = "/mnt/cephfs/dataset/pointclouds/p2p_visualization/nuscenes/ours0.01/debug-False_0802_id-val/plot/"

    for frame in ("pred_21_21", "pred_15_15", "pred_173_173"):
        our_file = os.path.join(our_path, frame + ".npy")
        salsa_file = our_file.replace(our_path, salsa_path)
        label_file = salsa_file.replace(
            "pred_", "label_"
        )  # .replace('salsanext0.1', 'salsanext0.01')
        points_file = salsa_file.replace(
            "pred_", "pcd_"
        )  # .replace('salsanext0.1', 'salsanext0.01')
        salsa = np.load(salsa_file)  # .reshape(-1)  # [:len(gt)]
        our = np.load(our_file)  # .reshape(-1)  # [:len(gt)]
        label = np.load(label_file)
        points = np.load(points_file)

        print(salsa_file)
        print(our_file)
        # save_ply(points, label, '{}_{}'.format(frame, 'label'), dir=save_plot_dir)
        save_ply(points, our, "{}_{}".format(frame, "our0.01"), dir=save_plot_dir)
        save_ply(points, salsa, "{}_{}".format(frame, "salsa0.01"), dir=save_plot_dir)
