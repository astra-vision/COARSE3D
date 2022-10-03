#!/usr/bin/env python3
# import sys
# sys.path.insert(0, "H:\\project\\202009Camera-lidar-segmentation\\poincloudProcessor")
import os
import open3d
from .common import getPointCloud, loadPCD
import numpy as np

from pc_processor.dataset.preprocess.augmentor import AugmentParams, Augmentor
import matplotlib.pyplot as plt


class SemanticKittiViewer(object):
    def __init__(self, dataset, show_img=False):
        self.dataset = dataset
        self.pointcloud_files = self.dataset.pointcloud_files
        self.n_samples = len(self.dataset)
        self.index = 0
        self.fig, self.ax = plt.subplots()
        # plt.axes("off")
        self.save_path = "./pointclouds/"
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

    def run(self):
        def update(vis):
            opt = vis.get_render_option()
            if os.path.isfile("option.json"):
                opt.load_from_json("option.json")

            pcd = loadPCD(self.pointcloud_files[self.index])
            print(self.pointcloud_files[self.index])

            sem_label, _ = self.dataset.readLabel(self.dataset.label_files[self.index])
            # pred_file = "H:\\project\\202009Camera-lidar-segmentation\\experiments\\PMF-semantickitti\\preds\\sequences\\08\\predictions\\{:06d}.label".format(self.index)
            # pred_label, _ = self.dataset.readLabel(pred_file)
            # print(sem_label)
            sem_label_map = self.dataset.labelMapping(sem_label)
            # pred_label[sem_label_map==0] = 0
            # sem_label_color = self.dataset.sem_color_lut_inv[self.dataset.labelMapping(sem_label)]
            sem_label_color = self.dataset.sem_color_lut[
                self.dataset.class_map_lut_inv[self.dataset.labelMapping(sem_label)]
            ]
            # sem_label_color = self.dataset.sem_color_lut[self.dataset.class_map_lut_inv[self.dataset.labelMapping(pred_label)]]
            # augmentor = Augmentor(params=AugmentParams)
            # pcd = augmentor.rotation(pcd, -75, -0, 90)
            # pcd = augmentor.translation(pcd, 40, 0, 0)
            pcd[sem_label_map == 0, :] = 0
            depth = np.linalg.norm(pcd[:, :3], axis=1)
            depth_map_color = plt.cm.jet(1 - depth / depth.max())
            # pointcloud = getPointCloud(pcd, sem_label_color)
            pointcloud = getPointCloud(pcd, depth_map_color[:, :3])
            vis.clear_geometries()
            vis.add_geometry(pointcloud)

            ctl = vis.get_view_control()
            param = open3d.io.read_pinhole_camera_parameters("./view.json")
            ctl.convert_from_pinhole_camera_parameters(param)
            vis.capture_screen_image(
                os.path.join(self.save_path, "{:06d}.jpeg".format(self.index))
            )
            # img = plt.imread("H:\\project\\202009Camera-lidar-segmentation\\paper_fig\\code\\visual_preds\\preds\\dense_preds\\{:06d}.jpeg".format(self.index))
            # self.ax.imshow(img)
            # plt.pause(0.1)
            # ctl.set_zoom(0.3)

        def moveForward(vis):
            print("move forward: ", self.index)
            self.index += 1
            if self.index >= self.n_samples:
                self.index = 0
            update(vis)
            return False

        def moveBackward(vis):
            print("move backward: ", self.index)
            self.index -= 1
            if self.index < 0:
                self.index = self.n_samples - 1
            update(vis)

            return False

        def saveOption(vis):
            opt = vis.get_render_option()
            opt.save_to_json("option.json")
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            open3d.io.write_pinhole_camera_parameters("view.json", param)
            # print(param)

        key_to_callback = {
            ord("N"): moveBackward,
            ord("M"): moveForward,
            ord("S"): saveOption,
        }
        print("==== How to use ===")
        print("Press S to save render options")
        print("Press + to increase point size")
        print("Press - to decrease point size")
        print("Press N to play previous frame")
        print("Press M to play next frame")

        pcd = loadPCD(self.pointcloud_files[self.index])
        pointcloud = getPointCloud(pcd)
        # img = open3d.io.read_image("H:\\project\\202009Camera-lidar-segmentation\\paper_fig\\code\\visual_preds\\preds\\dense_preds\\000000.jpeg")
        # img.
        # open3d.visualization.draw_geometries([img], point_show_normal=True)
        open3d.visualization.draw_geometries_with_key_callbacks(
            [pointcloud], key_to_callback, width=960, height=480
        )


# if __name__ == "__main__":
#     dataset = SemanticKitti(
#         root="h:\\dataset\\semantic-kitti\\dataset\\sequences\\",
#         sequences=[4],
#         config_path="H:\\project\\202009Camera-lidar-segmentation\\poincloudProcessor\\pc_processor\\dataset\\semantic_kitti\\config\\labels\\semantic-kitti.yaml"
#     )
#     viewer = SemanticKittiViewer(dataset)
#     viewer.run()
