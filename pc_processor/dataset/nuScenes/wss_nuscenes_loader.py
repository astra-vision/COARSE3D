import numpy as np
import torch
from torch.utils.data import Dataset

from pc_processor.dataset.preprocess import augmentor, projection


class SalsaNextLoader(Dataset):
    def __init__(
        self,
        dataset,
        config,
        data_len=-1,
        is_train=True,
        have_img=False,
        is_weak_label=True,
        use_cut_paste=False,
        return_uproj=False,
        use_mix3d=False,
        use_ground_mask=False,
        cuboid_erase=False,
        max_points=150000,  # max number of points present in dataset
        n_cls=17,
        debug=False,
    ):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.data_len = data_len
        self.return_uproj = return_uproj
        self.is_weak_label = is_weak_label
        self.use_cut_paste = use_cut_paste
        self.use_mix3d = use_mix3d
        self.mapped_cls_name = dataset.mapped_cls_name
        self.cuboid_erase = cuboid_erase
        self.have_img = have_img
        self.n_cls = n_cls
        self.max_points = max_points
        self.debug = debug

        if self.is_train:
            # 3d augmentator
            augment_params = augmentor.AugmentParams()
            augment_config = self.config["augmentation"]

            augment_params.setFlipProb(
                p_flipx=augment_config["p_flipx"], p_flipy=augment_config["p_flipy"]
            )
            augment_params.setTranslationParams(
                p_transx=augment_config["p_transx"],
                trans_xmin=augment_config["trans_xmin"],
                trans_xmax=augment_config["trans_xmax"],
                p_transy=augment_config["p_transy"],
                trans_ymin=augment_config["trans_ymin"],
                trans_ymax=augment_config["trans_ymax"],
                p_transz=augment_config["p_transz"],
                trans_zmin=augment_config["trans_zmin"],
                trans_zmax=augment_config["trans_zmax"],
            )
            augment_params.setRotationParams(
                p_rot_roll=augment_config["p_rot_roll"],
                rot_rollmin=augment_config["rot_rollmin"],
                rot_rollmax=augment_config["rot_rollmax"],
                p_rot_pitch=augment_config["p_rot_pitch"],
                rot_pitchmin=augment_config["rot_pitchmin"],
                rot_pitchmax=augment_config["rot_pitchmax"],
                p_rot_yaw=augment_config["p_rot_yaw"],
                rot_yawmin=augment_config["rot_yawmin"],
                rot_yawmax=augment_config["rot_yawmax"],
            )
            self.augmentor = augmentor.Augmentor(augment_params)

        else:
            self.augmentor = None
            self.new_label = None

        projection_config = self.config["sensor"]
        self.projection = projection.RangeProjection(
            fov_up=projection_config["fov_up"],
            fov_down=projection_config["fov_down"],
            fov_left=projection_config["fov_left"],
            fov_right=projection_config["fov_right"],
            proj_h=projection_config["proj_h"],
            proj_w=projection_config["proj_w"],
        )
        self.proj_img_mean = torch.tensor(
            self.config["sensor"]["img_mean"], dtype=torch.float
        )
        self.proj_img_stds = torch.tensor(
            self.config["sensor"]["img_stds"], dtype=torch.float
        )

    def __getitem__(self, index):

        pointcloud, sem_label, _, weak_label = self.dataset.loadDataByIndex(index)
        assert len(pointcloud) == len(sem_label) == len(weak_label), (
            "pcd length is {}, sem label length is {}, "
            "weak label length is {}, you could trace file like {} \n{} \n{}".format(
                len(pointcloud),
                len(sem_label),
                len(weak_label),
                self.dataset.pointcloud_files[index],
                self.dataset.label_files[index],
                self.dataset.weak_label_files[index],
            )
        )

        # map to [0-cls]
        sem_label = self.dataset.labelMapping(sem_label)  # n

        if weak_label.max() > self.n_cls:
            weak_label = self.dataset.labelMapping(weak_label)

        label_list = np.zeros(self.n_cls)

        if self.is_train:
            pointcloud = self.augmentor.doAugmentation(pointcloud)  # n, 4

        (
            proj_pointcloud,
            proj_range,
            proj_idx,
            proj_eval_mask,
        ) = self.projection.doProjection(pointcloud)

        proj_eval_label = np.zeros(
            (proj_eval_mask.shape[0], proj_eval_mask.shape[1]), dtype=np.float32
        )
        proj_eval_label[proj_idx > -1] = sem_label[proj_idx[proj_idx > -1]]

        proj_train_label = np.zeros(
            (proj_eval_mask.shape[0], proj_eval_mask.shape[1]), dtype=np.float32
        )
        proj_train_label[proj_idx > -1] = weak_label[proj_idx[proj_idx > -1]]

        # change to tensor
        proj_eval_label_tensor = torch.from_numpy(proj_eval_label)
        proj_eval_mask_tensor = torch.from_numpy(proj_eval_mask)
        proj_train_label_tensor = torch.from_numpy(proj_train_label)
        # proj_train_mask_tensor = torch.from_numpy(proj_train_mask)
        label_list_tensor = torch.from_numpy(label_list).long()

        # tar_frame is for plot entropy
        seq_id, frame_id = self.dataset.parsePathInfoByIndex(index)
        tar_frame = torch.Tensor([0]).bool()

        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])
        proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        proj_intensity_tensor = (
            proj_intensity_tensor.ne(-1).float() * proj_intensity_tensor
        )
        proj_feature_tensor = torch.cat(
            [
                proj_range_tensor.unsqueeze(0),
                proj_xyz_tensor.permute(2, 0, 1),
                proj_intensity_tensor.unsqueeze(0),
            ],
            0,
        )

        if self.return_uproj:
            uproj_x_idx = torch.from_numpy(
                self.projection.cached_data["uproj_x_idx"]
            )  # n,
            uproj_y_idx = torch.from_numpy(
                self.projection.cached_data["uproj_y_idx"]
            )  # n,
            uproj_depth = torch.from_numpy(
                self.projection.cached_data["uproj_depth"]
            )  # n,
            uproj_ref = torch.from_numpy(pointcloud[:, 3])  # n
            return (
                proj_feature_tensor,
                proj_train_label_tensor,
                proj_eval_label_tensor,
                proj_eval_mask_tensor,
                label_list,
                seq_id,
                frame_id,
                uproj_x_idx,
                uproj_y_idx,
                uproj_depth,
                uproj_ref,
            )

        # load unproj data
        unproj_n_points = len(pointcloud)

        unproj_full_labels = torch.full([self.max_points], 0, dtype=torch.int32)
        unproj_weak_labels = torch.full([self.max_points], 0, dtype=torch.int32)
        uproj_x_idx = torch.full([self.max_points], 0, dtype=torch.int32)
        uproj_y_idx = torch.full([self.max_points], 0, dtype=torch.int32)
        uproj_depth = torch.full([self.max_points], -1, dtype=torch.int32)
        uproj_ref = torch.full([self.max_points], -1, dtype=torch.int32)

        unproj_full_labels[:unproj_n_points] = torch.from_numpy(sem_label)
        unproj_weak_labels[:unproj_n_points] = torch.from_numpy(weak_label)
        uproj_x_idx[:unproj_n_points] = torch.from_numpy(
            self.projection.cached_data["uproj_x_idx"]
        )
        uproj_y_idx[:unproj_n_points] = torch.from_numpy(
            self.projection.cached_data["uproj_y_idx"]
        )
        uproj_depth[:unproj_n_points] = torch.from_numpy(
            self.projection.cached_data["uproj_depth"]
        )
        uproj_ref[:unproj_n_points] = torch.from_numpy(pointcloud[:, 3])

        if self.is_weak_label:
            return (
                proj_feature_tensor,
                proj_train_label_tensor,
                proj_eval_label_tensor,
                label_list_tensor,
                tar_frame,
                seq_id,
                frame_id,
                unproj_full_labels,
                unproj_weak_labels,
                uproj_x_idx,
                uproj_y_idx,
            )
        else:
            return (
                proj_feature_tensor,
                proj_eval_label_tensor,
                proj_eval_label_tensor,
                label_list_tensor,
                tar_frame,
                seq_id,
                frame_id,
                unproj_full_labels,
                unproj_weak_labels,
                uproj_x_idx,
                uproj_y_idx,
            )

    def __len__(self):
        if self.data_len > 0 and self.data_len < len(self.dataset):
            return self.data_len
        else:
            return len(self.dataset)
