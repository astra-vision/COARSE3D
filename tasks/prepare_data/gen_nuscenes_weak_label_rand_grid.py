import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch
import yaml

sys.path.insert(0, "../../")
import pc_processor

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']
EXTENSIONS_OCOC_LABEL = ['.npy']

os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def is_weak_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_OCOC_LABEL)


class NuscenesData():
    '''
    be suitable for semantic poss and semantic kitti
    '''

    def __init__(self,
                 dataset,
                 args,
                 ):
        self.dataset = nuscenes
        self.args = args
        self.mean_dxyzf = np.zeros(5)
        self.std_dxyzf = np.zeros(5)
        self.scan_count = 0

        self.class_pts_num = np.zeros(20)
        self.class_voxel_num = np.zeros(20)
        self.class_pts_num_full = np.zeros(20)

        # read config
        data_config = yaml.safe_load(open(args.data_config_path, "r"))
        self.mapped_class_name = data_config['mapped_class_name']

        self.label_map = np.zeros((360,)).astype(np.int32)
        for key in data_config['learning_map'].keys():
            self.label_map[key] = data_config['learning_map'][key]

        return

    def __len__(self):
        return len(self.dataset.token_list)

    def __getitem__(self, dataset_index):

        # load data
        scan, label, _, _ = self.dataset.loadDataByIndex(dataset_index)

        assert len(scan) == len(label)

        # # ----------------
        # # compute mean and std
        # # ----------------
        # depth = np.linalg.norm(scan[:, :3], 2, axis=1).reshape(-1, 1)
        # depth = np.minimum(depth, 200)
        # dxyzf = np.concatenate([depth, scan], -1)
        #
        # self.std_dxyzf = (dxyzf.std(0) + (self.scan_count * self.std_dxyzf)) / (self.scan_count + 1)
        # self.mean_dxyzf = (dxyzf.mean(0) + (self.scan_count * self.mean_dxyzf)) / (self.scan_count + 1)
        #
        # self.scan_count += 1
        # return self.mean_dxyzf, self.std_dxyzf
        # # ----------------

        mapped_label = self.label_map[label]  # [0 - 16]

        # grid sampling
        xyz = o3d.geometry.PointCloud()
        xyz.points = o3d.utility.Vector3dVector(scan[:, :3])

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(xyz,
                                                                    voxel_size=self.args.voxel_size)  # 60,325 => 12,237

        num_voxel = len(voxel_grid.get_voxels())
        prune_ratio = num_voxel / len(scan)
        if self.args.debug:
            print('prune to ', num_voxel, prune_ratio)

        # get voxel coord along the pcd index
        point2voxel = np.asarray([voxel_grid.get_voxel(pt) for pt in scan[:, :3]])  # (n_p, 3)
        voxels_coord, point2voxel_map, num_pts_in_voxel = np.unique(point2voxel, return_index=True, return_counts=True,
                                                                    axis=0)  # (n_v, 3) (n_v, n_v)

        voxel_label = mapped_label[point2voxel_map]  # (n_v, )
        assert len(np.unique(voxel_label)) > 1

        # compute voxel number need to label, e.g. point_number * 0.1%
        sample_voxel = int(np.around(len(scan) * self.args.label_ratio))  # 0.001 = 0.1% , 0.0001 = 0.01%

        # at least sample 1
        if sample_voxel < 1:
            sample_voxel = 1

        voxel_weak_label = np.zeros_like(voxel_label, dtype=label.dtype)  # (n_v, )
        point_weak_label = np.zeros_like(label, dtype=label.dtype)

        # select valid index
        valid_idxes = np.where(voxel_label > 0)[0]

        # select voxel id to sample
        sample_idx = np.random.choice(valid_idxes, sample_voxel, replace=False)

        # assign voxel label
        voxel_weak_label[sample_idx] = voxel_label[sample_idx]

        # get voxel id for sample
        voxel_sampled = voxels_coord[sample_idx]  # (k, 3)

        # enumarate each sampled voxel
        for i in range(len(voxel_sampled)):
            cls_voxel = voxel_weak_label[sample_idx][i]

            if args.voxel_propagation:
                pt_idx = (point2voxel == voxel_sampled[i]).min(1).nonzero()[0]
            else:
                pt_idx = (point2voxel == voxel_sampled[i]).min(1).nonzero()[0][0]

            point_weak_label[pt_idx] = cls_voxel

        num_labelled_pts = (point_weak_label > 0).sum()
        if args.debug:
            print('Label ratio {}, sampled voxels {}/{}, selected {}/{} ({:.8f}) points to label '
                  .format(self.args.label_ratio,
                          sample_voxel,
                          num_voxel,
                          num_labelled_pts,
                          len(scan),
                          num_labelled_pts / len(scan),
                          ))

        for cls in np.arange(20):
            if cls == 0:
                continue
            cls_pts = (point_weak_label == cls).sum()
            cls_voxels = (voxel_weak_label == cls).sum()

            cls_fully_labelled = (mapped_label == cls).sum()
            self.class_pts_num[cls] += cls_pts
            self.class_voxel_num[cls] += cls_voxels
            self.class_pts_num_full[cls] += cls_fully_labelled

        # file name
        label_file = self.dataset.loadLabelPathByIndex(dataset_index)
        # e.g.
        # /mnt/cephfs/dataset/pointclouds/nuscenes/lidarseg/v1.0-mini/fdddd75ee1d94f14a09991988dab8b3e_lidarseg.bin

        new_weak_label_file = label_file.replace(self.args.dataset_root, self.args.dataset_save). \
            replace('lidarseg', self.args.dir_save).replace('.bin', '.npy')
        # e.g.
        # /mnt/cephfs/dataset/pointclouds/nuscenes-coarse3d//xxx/v1.0-mini/fdddd75ee1d94f14a09991988dab8b3e_xxx.npy

        # make dir to save
        os.makedirs(new_weak_label_file.replace(os.path.basename(new_weak_label_file), '/'), exist_ok=True)
        # e.g.
        # /mnt/cephfs/dataset/pointclouds/nuscenes-coarse3d/xxx/v1.0-mini/

        return point_weak_label, new_weak_label_file, dataset_index, sample_voxel, num_labelled_pts, \
               self.class_pts_num, self.class_voxel_num, self.class_pts_num_full, self.std_dxyzf, self.mean_dxyzf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--debug',
                        # default=True,
                        default=False,
                        help='if use debug mode, it is a no multi process realization')
    parser.add_argument('--dataset',
                        default='nuScenes',
                        help=''
                        )
    parser.add_argument('--dataset_root',
                        default='/mnt/cephfs/dataset/pointclouds/nuscenes',
                        help=''
                        )
    parser.add_argument('--dataset_save',
                        default='/mnt/cephfs/dataset/pointclouds/nuscenes-coarse3d/',
                        help='the path where you save the weak label, do not recommend to set it same as dataset_root'
                        )
    parser.add_argument('--data_config_path',
                        default='../../pc_processor/dataset/nuScenes/nuscenes.yaml',
                        help='dataset config file'
                        )
    parser.add_argument('--version',
                        # default='v1.0-mini',  # this is for debug
                        default='v1.0-trainval',
                        )
    parser.add_argument('--split',
                        # default='val',
                        default='train',
                        help='you need to run under both `train` and `val` mode',
                        )

    parser.add_argument('--weak_label_name',
                        default='0.1',
                        help='the dir name of generated weak label',
                        )
    parser.add_argument('--label_ratio',
                        default=0.001,
                        help='0.001=>0.1%, 0.0001=>0.01%,  0.01=>1%'
                        )
    parser.add_argument('--voxel_size',
                        default=0.06,
                        type=float,
                        )
    parser.add_argument('--voxel_propagation',
                        default=True,
                        )

    args = parser.parse_args()

    nuscenes = pc_processor.dataset.nuScenes.Nuscenes(root=[args.dataset_root, args.dataset_root],
                                                      version=args.version,
                                                      split=args.split,
                                                      return_ref=False,
                                                      config_path=args.data_config_path,
                                                      )

    dataset = NuscenesData(dataset=nuscenes, args=args)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=1,
                                               num_workers=0 if args.debug else 60,
                                               drop_last=False,
                                               shuffle=False,
                                               )

    cnt_voxels = 0  # labelled voxels
    cnt_pts = 0  # labelled points
    point_num = 0  # lidar point number
    cnt_class_pts_num = []
    cnt_class_voxel_num = []
    cnt_class_pts_num_full = []
    mean_feat = 0
    std_feat = 0

    for i, (weak, weak_file, current_index, sample_voxel, num_labelled_pts, class_pts_num,
            class_voxel_num, class_pts_num_full, std_, mean_) in enumerate(train_loader):
        print('{}/{} save {}'.format(current_index[0], len(dataset), weak_file[0]))

        np.save(weak_file[0], weak)

        cnt_voxels = cnt_voxels + np.asarray(sample_voxel[0])
        cnt_pts = cnt_pts + np.asarray(num_labelled_pts[0])
        point_num = point_num + np.asarray(len(weak[0]))
        mean_feat = mean_
        std_feat = std_

        cnt_class_pts_num = np.asarray(class_pts_num[0])
        cnt_class_voxel_num = np.asarray(class_voxel_num[0])
        cnt_class_pts_num_full = np.asarray(class_pts_num_full[0])

        if args.debug:
            break

    # log
    with open('./log_{}_ratio-{}_voxel_-{}_prop-{}.txt'.
                      format(args.dataset, args.label_ratio, args.voxel_size, args.voxel_propagation), "w") as f:

        f.write("\n\n\n{} \n".format('*' * 20))

        for i in args._get_kwargs():
            f.write('{} \n'.format(i))

        f.write("\n\n\n{} \n".format('*' * 20))
        f.write("per class voxels \n")
        f.write("{} \n".format('*' * 20))

        for cls in np.arange(len(cnt_class_pts_num)):
            f.write('{}: {}\n'.format(cls, cnt_class_voxel_num[cls]))

        # f.write("\n\n\n{} \n".format('*' * 20))

        # f.write('labelled voxels {} \n'.format(cnt_voxels))
        # f.write('labelled pts {} \n'.format(cnt_pts))
        # f.write('appeared pts {} \n'.format(point_num))
        # f.write('mean  {} \n'.format(mean_feat))
        # f.write('std {} \n'.format(std_feat))

        # f.write("\n\n\n{} \n".format('*' * 20))
        # f.write("per class points \n")
        # f.write("{} \n".format('*' * 20))
        #
        # for cls in np.arange(len(cnt_class_pts_num)):
        #     f.write('{}: {}\n'.format(cls, cnt_class_pts_num[cls]))

        # f.write("\n\n\n{} \n".format('*' * 20))
        # f.write("per class points (fully sup) \n")
        # f.write("{} \n".format('*' * 20))
        #
        # for cls in np.arange(len(cnt_class_pts_num)):
        #     f.write('{}: {}\n'.format(cls, cnt_class_pts_num_full[cls]))
