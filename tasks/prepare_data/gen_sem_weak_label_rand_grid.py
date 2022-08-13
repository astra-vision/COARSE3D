# from pc_processor.visualizer.vis_as_ply import save_ply
import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch
import yaml

sys.path.append('..')

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


class SemanticData():
    '''
    be suitable for semantic poss and semantic kitti
    '''

    def __init__(self,
                 args,
                 ):
        self.args = args
        self.mean_dxyzf = np.zeros(5)
        self.std_dxyzf = np.zeros(5)
        self.scan_count = 0

        # read config
        data_config = yaml.safe_load(open(args.data_config_path, "r"))
        self.mapped_class_name = data_config['mapped_class_name']

        self.label_map = np.zeros((360,)).astype(np.int32)
        for key in data_config['learning_map'].keys():
            # print(key, label_map[key])
            self.label_map[key] = data_config['learning_map'][key]

        # read data
        self.scan_files = {}
        self.label_files = {}
        self.weak_label_files = {}
        self.dataset_size = 0
        self.index_mapping = {}
        self.poses = {}
        self.proj_matrix = {}
        self.class_pts_num = np.zeros(20)
        self.class_voxel_num = np.zeros(20)
        self.class_pts_num_full = np.zeros(20)
        dataset_index = 0

        for seq in self.args.sequences:
            # to string
            seq = '{0:02d}'.format(int(seq))
            print("parsing seq {}".format(seq))

            # get paths for each
            scan_path = os.path.join(self.args.dataset_root, seq, "velodyne")
            label_path = os.path.join(self.args.dataset_root, seq, "labels")

            assert os.path.exists(scan_path)
            assert os.path.exists(label_path)

            # create save path for generated weak label
            os.makedirs(os.path.join(self.args.dataset_save, seq, self.args.dir_save), exist_ok=True)

            # get files
            scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
            print(scan_path)
            label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(label_path)) for f in fn if is_label(f)]

            scan_files.sort()
            label_files.sort()
            assert (len(scan_files) == len(label_files))

            self.scan_files[seq] = scan_files
            self.label_files[seq] = label_files

            for frame_id in range(len(scan_files)):
                self.index_mapping[dataset_index] = (seq, frame_id)
                dataset_index += 1
            self.dataset_size += len(scan_files)

            print("Using {} scans from sequences".format(self.dataset_size))

        # end sequences

        return

    def __len__(self):
        return self.dataset_size

    def load_scan(self, filename):
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan

    def load_label(self, filename):
        if '.label' in filename:
            label = np.fromfile(filename, dtype=np.int32)
            label = label.reshape((-1))
            sem_label = label & 0xFFFF  # semantic label in lower half
            # inst_label = label >> 16  # instance id in upper half
            return sem_label
        elif '.npy' in filename:
            label = np.load(filename)
            return label
        else:
            raise IOError('Neither weak label nor semantic label is found')

    def __getitem__(self, dataset_index):

        # load data
        seq, current_index = self.index_mapping[dataset_index]

        # read current frame
        scan_file = self.scan_files[seq][current_index]
        label_file = self.label_files[seq][current_index]

        scan = self.load_scan(scan_file)
        label = self.load_label(label_file)

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

        mapped_label = (self.label_map[label])  # [0 - 19] in semantic kitti, or [0 - 13] in semantic poss

        new_weak_label_file = label_file.replace(self.args.dataset_root, self.args.dataset_save). \
            replace('labels', self.args.dir_save).replace('.label', '.npy')

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
        voxels_coord, point2voxel_map, num_pts_in_voxel = np.unique(point2voxel,
                                                                    return_index=True,
                                                                    return_counts=True,
                                                                    axis=0)

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

        if self.args.debug:
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

        return point_weak_label, new_weak_label_file, current_index, sample_voxel, num_labelled_pts, \
               self.class_pts_num, self.class_voxel_num, self.class_pts_num_full


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--debug',
                        # default=True,
                        default=False,
                        help='if use debug mode, it is a no multi process realization')
    parser.add_argument('--dataset',
                        default='SemanticKITTI',
                        # default='SemanticPOSS',
                        help='interested dataset'
                        )
    parser.add_argument('--dataset_root',
                        # default='/mnt/cephfs/dataset/pointclouds/semantic-kitti/dataset/sequences/',
                        # default='/mnt/cephfs/dataset/pointclouds/semantic-poss/dataset/sequences',
                        help='dataset root'
                        )
    parser.add_argument('--dataset_save',
                        # default='/mnt/cephfs/dataset/pointclouds/semantic-kitti-coarse3d/sequences/',
                        default='/mnt/cephfs/dataset/pointclouds/semantic-poss-coarse3d/sequences/',
                        help='the path where you save the weak label, do not recommend to set it same as dataset_root'
                        )
    parser.add_argument('--data_config_path',
                        # default='../../pc_processor/dataset/semantic_kitti/semantic-kitti.yaml',
                        default='../../pc_processor/dataset/semantic_poss/semantic-poss.yaml',
                        help='dataset config file'
                        )
    parser.add_argument('--sequences',
                        default=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),  # Semantic KITTI
                        # default=(0, 1, 2, 3, 4, 5),  # Semantic POSS
                        type=tuple,
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

    dataset = SemanticData(args=args)

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

    # for i, (mean_value, std_value) in enumerate(train_loader):
    #     print(i, " / ", len(train_loader))
    #
    # print("\n\n\n{} \n".format('*' * 20))
    # print('mean dxyzf is {}\n\n'.format(mean_value))
    # print('std dxyzf is {}'.format(std_value))

    for i, (weak, weak_file, current_index, sample_voxel, num_labelled_pts, class_pts_num,
            class_voxel_num, class_pts_num_full) in enumerate(train_loader):

        print('save ', weak_file[0])
        np.save(weak_file[0], weak)

        cnt_voxels = cnt_voxels + np.asarray(sample_voxel[0])
        cnt_pts = cnt_pts + np.asarray(num_labelled_pts[0])
        point_num = point_num + np.asarray(len(weak[0]))

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
        #
        # f.write('labelled voxels {} \n'.format(cnt_voxels))
        # f.write('labelled pts {} \n'.format(cnt_pts))
        # f.write('appeared pts {} \n'.format(point_num))

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
        #     f.write(
        #         '{}: {}\n'.format(cls, cnt_class_pts_num_full[cls]))
