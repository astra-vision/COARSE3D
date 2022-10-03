import os

import numpy as np
import yaml


class SemanticPOSS(object):
    def __init__(
        self,
        root,  # directory where data is #  [pcd_root, weak_root]
        sequences,  # sequences for this data (e.g. [1,3,4,6])
        config_path,  # directory of config file
        has_weak_label=False,
        weak_label_name="",
        has_label=True,
        range_h=40,  # height of range image
        range_w=1800,  # width of range image
    ):
        self.root = root  #
        self.sequences = sequences
        self.sequences.sort()  # sort seq id
        self.has_label = has_label
        self.has_weak_label = has_weak_label
        self.weak_label_name = weak_label_name
        self.proj_h = range_h
        self.proj_w = range_w
        # check file exists
        if os.path.isfile(config_path):
            self.data_config = yaml.safe_load(open(config_path, "r"))
        else:
            raise ValueError("config file not found: {}".format(config_path))

        for i in self.root:
            if os.path.isdir(i):
                print("Dataset found: {}".format(i))
            else:
                raise ValueError("Dataset not found: {}".format(i))

        assert os.path.exists(self.root[0]), ValueError(
            "Dataset not found: {}".format(self.root[0])
        )

        if self.has_weak_label:
            assert os.path.exists(self.root[1]), ValueError(
                "Dataset not found: {}".format(self.root[1])
            )

        self.pointcloud_files = []
        self.tag_files = []
        self.label_files = []
        self.weak_label_files = []
        self.proj_matrix = {}

        for seq in self.sequences:
            # format seq id
            seq = "{0:02d}".format(int(seq))
            print("parsing seq {}...".format(seq))

            # get file list from path
            pointcloud_path = os.path.join(self.root[0], seq, "velodyne")
            pointcloud_files = [
                os.path.join(pointcloud_path, f)
                for f in os.listdir(pointcloud_path)
                if ".bin" or ".npy" in f
            ]
            self.pointcloud_files.extend(pointcloud_files)

            #  File XXXXXX.tag in the tag folder is used for generating range image,
            #  which records the position of each point in range image.
            tag_path = os.path.join(self.root[0], seq, "tag")
            tag_files = [
                os.path.join(tag_path, f) for f in os.listdir(tag_path) if ".tag" in f
            ]
            self.tag_files.extend(tag_files)
            assert len(pointcloud_files) == len(tag_files)

            if self.has_label:
                label_path = os.path.join(self.root[0], seq, "labels")
                label_files = [
                    os.path.join(label_path, f)
                    for f in os.listdir(label_path)
                    if ".label" in f
                ]
                assert len(pointcloud_files) == len(label_files)
                self.label_files.extend(label_files)

            if self.has_weak_label:
                label_path = os.path.join(self.root[1], seq, self.weak_label_name)
                weak_label_files = [
                    os.path.join(label_path, f)
                    for f in os.listdir(label_path)
                    if ".label" or ".npy" in f
                ]
                self.weak_label_files.extend(weak_label_files)
                assert len(pointcloud_files) == len(weak_label_files)

        # sort for correspondance
        assert len(self.pointcloud_files) > 0, "no point cloud file is found !!!"

        self.pointcloud_files.sort()
        self.tag_files.sort()

        if self.has_label:
            self.label_files.sort()
        if self.has_weak_label:
            self.weak_label_files.sort()
        print(
            "Using {} pointclouds from sequences {}".format(
                len(self.pointcloud_files), self.sequences
            )
        )

        # load config
        # get color map
        sem_color_map = self.data_config["color_map"]  # (23, 3)
        max_sem_key = 0
        for k, v in sem_color_map.items():
            if k + 1 > max_sem_key:
                max_sem_key = k + 1
        self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
        for k, v in sem_color_map.items():
            self.sem_color_lut[k] = np.array(v, np.float32) / 255.0

        sem_color_inv_map = self.data_config["color_map_inv"]  # 14, 3
        self.sem_colormap = np.zeros((14, 3), dtype=np.float32)
        for k, v in sem_color_inv_map.items():
            self.sem_colormap[k] = np.array(v, np.float32) / 255.0

        max_sem_key = 0
        for k, v in sem_color_inv_map.items():
            if k + 1 > max_sem_key:
                max_sem_key = k + 1
        self.sem_color_lut_inv = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
        for k, v in sem_color_inv_map.items():
            self.sem_color_lut_inv[k] = np.array(v, np.float32) / 255.0

        self.inst_color_map = np.random.uniform(low=0.0, high=1.0, size=(10000, 3))

        # get learning class map
        # map unused classes to used classes
        learning_map = self.data_config["learning_map"]
        max_key = 0
        for k, v in learning_map.items():
            if k > max_key:
                max_key = k
        # +100 hack making lut bigger just in case there are unknown labels
        self.class_map_lut = np.zeros((max_key + 100), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut[k] = v
        # learning map inv
        learning_map = self.data_config["learning_map_inv"]
        max_key = 0
        for k, v in learning_map.items():
            if k > max_key:
                max_key = k
        # +100 hack making lut bigger just in case there are unknown labels
        self.class_map_lut_inv = np.zeros((max_key + 100), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut_inv[k] = v

        self.mapped_cls_name = self.data_config["mapped_class_name"]

    def read_weak_label(self, filename):
        label = np.load(filename).reshape(-1)
        return label

    def get_rangeimage(self, index, pointcloud, full_label, weak_label):
        tags = self.loadTagByIndex(index)
        assert tags.sum() == len(pointcloud)

        # get depth of all points
        depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1)
        depth = np.minimum(depth, 200)

        proj_range = np.full((self.proj_h * self.proj_w), -1, dtype=np.float32)
        proj_range[tags] = depth
        proj_range = np.reshape(proj_range, (self.proj_h, self.proj_w))

        proj_pointcloud = np.full(
            (self.proj_h * self.proj_w, pointcloud.shape[1]), -1, dtype=np.float32
        )
        proj_pointcloud[tags] = pointcloud
        proj_pointcloud = np.reshape(
            proj_pointcloud, (self.proj_h, self.proj_w, pointcloud.shape[1])
        )

        proj_full_label = np.full((self.proj_h * self.proj_w), 0, dtype=np.int32)
        proj_full_label[tags] = full_label
        proj_full_label = np.reshape(proj_full_label, (self.proj_h, self.proj_w))

        proj_weak_label = np.full((self.proj_h * self.proj_w), 0, dtype=np.int32)
        proj_weak_label[tags] = weak_label
        proj_weak_label = np.reshape(proj_weak_label, (self.proj_h, self.proj_w))

        proj_mask = (proj_range > -1).astype(np.bool)
        assert len(pointcloud) == proj_mask.sum()

        return (
            proj_pointcloud,
            proj_range,
            proj_mask,
            proj_weak_label,
            proj_full_label,
            tags,
            depth,
        )

    @staticmethod
    def readPCD(path):
        if ".npy" in path:
            pcd = np.load(path)
        else:
            pcd = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return pcd

    @staticmethod
    def readLabel(path):
        if ".npy" in path:
            sem_label = np.load(path)
            inst_label = []
        else:
            label = np.fromfile(path, dtype=np.int32)
            sem_label = label & 0xFFFF  # semantic label in lower half
            inst_label = label >> 16  # instance id in upper half
        return sem_label, inst_label

    def parsePathInfoByIndex(self, index):
        path = self.pointcloud_files[index]
        path_split = path.split("/")
        seq_id = path_split[-3]
        frame_id = path_split[-1].split(".")[0]
        return seq_id, frame_id

    def labelMapping(self, label):
        label = self.class_map_lut[label]
        return label

    def loadTagByIndex(self, index):
        tag_file = self.tag_files[index]
        tags = np.fromfile(tag_file, dtype=np.bool)
        return tags

    def loadLabelByIndex(self, index):
        sem_label, inst_label = self.readLabel(self.label_files[index])
        return sem_label, inst_label

    def loadDataByIndex(self, index):
        pointcloud = self.readPCD(self.pointcloud_files[index])
        if self.has_label:
            sem_label, inst_label = self.readLabel(self.label_files[index])
        else:
            sem_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
            inst_label = np.zeros(pointcloud.shape[0], dtype=np.int32)

        if self.has_weak_label:
            weak_label = self.read_weak_label(self.weak_label_files[index])
        else:
            weak_label = np.zeros(pointcloud.shape[0], dtype=np.int32)

        return pointcloud, sem_label, inst_label, weak_label

    def __len__(self):
        return len(self.pointcloud_files)
