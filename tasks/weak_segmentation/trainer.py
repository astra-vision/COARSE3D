import datetime
import os
import time

import cv2
import numpy as np

# import pydensecrf.densecrf as dcrf
import torch
import torch.nn as nn
import torch.nn.functional as F
from option import Option

import pc_processor


class Trainer(object):
    def __init__(self, settings: Option, model: nn.Module, recorder=None):
        # init params
        self.settings = settings
        self.recorder = recorder
        self.model = model.cuda()
        self.remain_time = pc_processor.utils.RemainTime(self.settings.n_epochs)

        self.labelled_points = 0
        self.points = 0

        # init data loader
        (
            self.train_loader,
            self.val_loader,
            self.train_sampler,
            self.val_sampler,
        ) = self._initDataloader()

        # init criterion
        self.criterion = self._initCriterion()

        # init optimizer
        self.optimizer = self._initOptimizer()
        self.scheduler = self._initScheduler()

        # load ckpt and pretrain
        self.model = self._loadCheckpoint(
            self.model,
            only_encoder=self.settings.only_encoder,
            pretrain_path=self.settings.pretrained_model,
        )

        # set multi gpu
        if self.settings.distributed:
            # sync bn
            local_rank = int(os.environ["LOCAL_RANK"])
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )

        self.evaluator = pc_processor.metrics.IOUEval(
            n_classes=self.settings.n_classes,
            device=torch.device("cuda"),
            ignore=self.ignore_class,
        )
        self.evaluator.reset()

    def _loadCheckpoint(self, model, only_encoder=False, pretrain_path=None):

        assert (
            pretrain_path is None or self.settings.checkpoint is None
        ), "cannot use pretrained weight and checkpoint at the same time"
        if pretrain_path is not None:
            if not os.path.isfile(pretrain_path):
                raise FileNotFoundError(
                    "pretrained model not found: {}".format(pretrain_path)
                )
            state_dict = torch.load(pretrain_path, map_location="cpu")

            if "model" in state_dict.keys():
                state_dict = state_dict["model"]
            else:
                assert "model_state" in state_dict.keys()
                state_dict = state_dict["model_state"]

            new_state_dict = model.state_dict()

            for k, v in state_dict.items():
                if k in new_state_dict.keys():
                    if only_encoder:
                        if not k in self.settings.encoder_modules:
                            # print('Not a encoder params ', k)
                            continue
                    if new_state_dict[k].size() == v.size():
                        new_state_dict[k] = v
                    else:
                        print("diff size: ", k, v.size())
                else:
                    print("diff key: ", k)

            model.load_state_dict(new_state_dict)
            if self.recorder is not None:
                self.recorder.logger.info(
                    "loading pretrained weight from: {}".format(pretrain_path)
                )

        if self.settings.checkpoint is not None:
            if not os.path.isfile(self.settings.checkpoint):
                raise FileNotFoundError(
                    "checkpoint file not found: {}".format(self.settings.checkpoint)
                )

            checkpoint_data = torch.load(self.settings.checkpoint, map_location="cpu")

            if "model" in checkpoint_data.keys():
                model.load_state_dict(checkpoint_data["model"])
            else:
                assert "model_state" in checkpoint_data.keys()
                new_state_dict = model.state_dict()
                for k, v in checkpoint_data["model_state"].items():
                    if k in new_state_dict.keys():
                        if new_state_dict[k].size() == v.size():
                            new_state_dict[k] = v
                    else:
                        print("diff size: ", k)
                model.load_state_dict(new_state_dict)

            self.optimizer.load_state_dict(checkpoint_data["optimizer"])
            self.scheduler.load_state_dict(checkpoint_data["scheduler"])
            self.epoch_start = checkpoint_data["epoch"]
            print("Successfully loaded ckpt from {}".format(self.settings.checkpoint))
        return model

    def _initScheduler(self):
        scheduler = pc_processor.utils.WarmupCosineLR(
            optimizer=self.optimizer,
            lr=self.settings.lr,
            warmup_steps=self.settings.warmup_epochs * len(self.train_loader),
            momentum=self.settings.momentum,
            max_steps=len(self.train_loader)
            * (self.settings.n_epochs - self.settings.warmup_epochs),
        )
        return scheduler

    def _initOptimizer(self):
        if self.settings.optimizer == "Adam":
            optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=self.settings.lr,
            )
        else:
            raise ValueError("invalid optimizer: {}".format(self.settings.optimizer))

        return optimizer

    def _initDataloader(self):
        if self.settings.dataset == "SemanticKitti":
            trainset = pc_processor.dataset.semantic_kitti.SemanticKitti(
                root=[self.settings.pcd_root, self.settings.weak_root],
                sequences=self.settings.train_seq,
                config_path=self.settings.data_config_path,
                has_weak_label=True,
                weak_label_name=self.settings.weak_label_name,
            )

            valset = pc_processor.dataset.semantic_kitti.SemanticKitti(
                root=[self.settings.pcd_root, self.settings.weak_root],
                sequences=self.settings.val_seq,
                config_path=self.settings.data_config_path,
                has_weak_label=False,
                weak_label_name=self.settings.weak_label_name,
            )

            train_salsa_loader = pc_processor.dataset.semantic_kitti.wss_sem_kitti_loader.SalsaNextLoader(
                dataset=trainset,
                config=self.settings.config,
                is_train=True,
                is_weak_label=self.settings.weak_label,
                n_cls=self.settings.n_classes,
            )

            val_salsa_loader = pc_processor.dataset.semantic_kitti.wss_sem_kitti_loader.SalsaNextLoader(
                dataset=valset,
                config=self.settings.config,
                is_train=False,
                is_weak_label=self.settings.weak_label,
                n_cls=self.settings.n_classes,
            )
        elif self.settings.dataset == "SemanticPOSS":
            trainset = pc_processor.dataset.semantic_poss.SemanticPOSS(
                root=[self.settings.pcd_root, self.settings.weak_root],
                sequences=self.settings.train_seq,
                config_path=self.settings.data_config_path,
                has_weak_label=True,
                weak_label_name=self.settings.weak_label_name,
            )

            valset = pc_processor.dataset.semantic_poss.SemanticPOSS(
                root=[self.settings.pcd_root, self.settings.weak_root],
                sequences=self.settings.val_seq,
                config_path=self.settings.data_config_path,
                has_weak_label=False,
                weak_label_name=self.settings.weak_label_name,
            )

            train_salsa_loader = (
                pc_processor.dataset.semantic_poss.wss_sem_poss_loader.SalsaNextLoader(
                    dataset=trainset,
                    config=self.settings.config,
                    is_train=True,
                    is_weak_label=self.settings.weak_label,
                    n_cls=self.settings.n_classes,
                )
            )

            val_salsa_loader = (
                pc_processor.dataset.semantic_poss.wss_sem_poss_loader.SalsaNextLoader(
                    dataset=valset,
                    config=self.settings.config,
                    is_train=False,
                    is_weak_label=self.settings.weak_label,
                    n_cls=self.settings.n_classes,
                )
            )
        elif self.settings.dataset == "nuScenes":
            trainset = pc_processor.dataset.nuScenes.Nuscenes(
                root=[self.settings.pcd_root, self.settings.weak_root],
                version="v1.0-trainval",
                # version='v1.0-mini',
                split="train",
                has_weak_label=True,
                filter_min_depth=True,
                config_path=self.settings.data_config_path,
                weak_label_name=self.settings.weak_label_name,
            )
            valset = pc_processor.dataset.nuScenes.Nuscenes(
                root=[self.settings.pcd_root, self.settings.weak_root],
                version="v1.0-trainval",
                # version='v1.0-mini',
                split="val",
                has_weak_label=False,
                filter_min_depth=True,
                config_path=self.settings.data_config_path,
                weak_label_name=self.settings.weak_label_name,
            )

            train_salsa_loader = (
                pc_processor.dataset.nuScenes.wss_nuscenes_loader.SalsaNextLoader(
                    dataset=trainset,
                    config=self.settings.config,
                    is_train=True,
                    is_weak_label=self.settings.weak_label,
                    n_cls=self.settings.n_classes,
                    debug=self.settings.is_debug,
                )
            )

            val_salsa_loader = (
                pc_processor.dataset.nuScenes.wss_nuscenes_loader.SalsaNextLoader(
                    dataset=valset,
                    config=self.settings.config,
                    is_train=False,
                    is_weak_label=self.settings.weak_label,
                    n_cls=self.settings.n_classes,
                    debug=self.settings.is_debug,
                )
            )
        else:
            raise ValueError("invalid dataset: {}".format(self.settings.dataset))

        # config
        cls_counts = np.asarray(
            [i for i in self.settings.config["cls_counts"].values()]
        )
        assert len(cls_counts) == self.settings.n_classes

        sum_counts = cls_counts.sum()
        cls_freq = np.asarray([i / sum_counts for i in cls_counts])

        self.cls_weight = 1 / (cls_freq + 1e-3)

        self.ignore_class = []
        for cl, _ in enumerate(self.cls_weight):
            if trainset.data_config["learning_ignore"][cl]:
                self.cls_weight[cl] = 0
            if self.cls_weight[cl] < 1e-10:
                self.ignore_class.append(cl)

        if self.recorder is not None:
            self.recorder.logger.info("weight: {}".format(self.cls_weight))
        self.mapped_cls_name = trainset.mapped_cls_name

        # extract colormap for visualization
        self.colormap = torch.from_numpy(valset.sem_color_lut).float().to("cpu")
        self.colormap = self.colormap * 255

        self.class_map_lut_inv = valset.class_map_lut_inv

        if self.settings.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset, shuffle=True, drop_last=True
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                valset, shuffle=False, drop_last=False
            )
            train_loader = torch.utils.data.DataLoader(
                train_salsa_loader,
                batch_size=self.settings.batch_size[0],
                num_workers=self.settings.n_threads,
                drop_last=True,
                sampler=train_sampler,
            )

            val_loader = torch.utils.data.DataLoader(
                val_salsa_loader,
                batch_size=self.settings.batch_size[1],
                num_workers=self.settings.n_threads,
                drop_last=False,
                sampler=val_sampler,
            )
            return train_loader, val_loader, train_sampler, val_sampler

        else:
            train_loader = torch.utils.data.DataLoader(
                train_salsa_loader,
                batch_size=self.settings.batch_size[0],
                num_workers=self.settings.n_threads,
                shuffle=True,
                drop_last=True,
            )

            val_loader = torch.utils.data.DataLoader(
                val_salsa_loader,
                batch_size=self.settings.batch_size[1],
                num_workers=self.settings.n_threads,
                shuffle=False,
                drop_last=False,
            )
            return train_loader, val_loader, None, None

    @staticmethod
    def get_rank():
        if not torch.distributed.is_initialized():
            return 0
        return torch.distributed.get_rank()

    def _initCriterion(self):
        criterion = {}

        alpha = np.log(1 + self.cls_weight)
        alpha = alpha / alpha.max()

        alpha[0] = 0
        if self.recorder is not None:
            self.recorder.logger.info("focal_loss alpha: {}".format(alpha))

        criterion["focal_loss"] = pc_processor.loss.FocalSoftmaxLoss(
            self.settings.n_classes, gamma=2, alpha=alpha, softmax=False
        )

        criterion["lovasz"] = pc_processor.loss.Lovasz_softmax(
            ignore=self.settings.ignore_cls, per_image=False, softmax=False
        )

        criterion["contrast_loss"] = pc_processor.loss.ContrastMEMLoss(
            ignore_label=self.settings.ignore_cls,
            temperature=self.settings.temperature,
            num_anchor=self.settings.num_anchor,
            is_debug=self.settings.is_debug,
        )

        # set device
        for _, v in criterion.items():
            v.cuda()

        return criterion

    def _combineTensorboradImages(
        self,
        argmax: torch.Tensor,
        full_label: torch.Tensor,
        weak_label: torch.Tensor,
        is_weak,
    ):
        assert argmax.ndim == 2
        assert argmax.shape == full_label.shape == weak_label.shape

        mask1 = full_label > 0  # h, w

        # recover to original label
        weak_label = self.class_map_lut_inv[weak_label]
        full_label = self.class_map_lut_inv[full_label]

        argmax = self.class_map_lut_inv[argmax]

        # white => black, note this is must, cuz the cv2 dilate operate
        self.colormap[0] = torch.Tensor([0, 0, 0])

        # colorization prediction
        color_argmax = self.colormap[argmax]  # h, w, 3

        # colorization fully label
        color_full_label = self.colormap[full_label]  # h, w, 3

        # vis error map
        mask2 = full_label != argmax
        mask = mask1 * mask2  # h, w
        error_map = (mask * 255).unsqueeze(-1).repeat(1, 1, 3)  # h, w, 3

        # colorization weakly label
        color_weak_label = (np.array(self.colormap[weak_label])).astype(
            np.uint8
        )  # h, w, 3, [0,255]

        if is_weak:
            # dilate
            color_weak_label = cv2.dilate(
                color_weak_label, np.ones((5, 5), np.uint8), iterations=1
            )  # h, w, 3, [0,255]

        color_weak_label = torch.from_numpy(color_weak_label).permute(
            2, 0, 1
        )  # 3, h, w

        # concat

        result = torch.cat(
            [
                color_weak_label / 255,
                (color_argmax.permute(2, 0, 1)) / 255,
                (color_full_label.permute(2, 0, 1)) / 255,
                (error_map.permute(2, 0, 1)) / 255,
            ],
            1,
        )  # 3, h*4, w

        # save
        # -----
        # from PIL import Image
        # a = Image.fromarray(np.array((result * 255).permute(1, 2, 0)).astype(np.uint8))
        # a.save("/mnt/cephfs/home/lirong/code/poincloudProcessor-P2PContrast/proto_{}.jpg".format(frame))
        # -----

        return result

    def entropy_based_selection(
        self,
        output,
        wss_mask,
        eval_mask,
        train_label,
        select_ratio,
    ):

        bs, _, h, w = output.shape

        # compute pcd entropy: p * log p
        entropy = -torch.sum(
            output * torch.log(output + 1e-10), dim=1
        )  # b, h, w  [0, ]

        _, pseudo_label = torch.max(output, dim=1)  # b, h, w

        # compute sample weight
        entropy_weights = torch.exp(-1 * entropy)  # normalize (0, 1]

        # filter ignore pixel
        pseudo_label[eval_mask == False] = self.settings.ignore_cls

        low_entropy_mask = torch.zeros(bs, self.settings.n_classes, h, w).bool().cuda()

        for b in range(bs):
            unique_class = torch.unique(train_label[b])

            for cls in unique_class:
                if cls == self.settings.ignore_cls:
                    continue

                cls_mask = (pseudo_label[b] == cls) * (eval_mask[b] > 0)  # h, w

                if cls_mask.sum() == 0:
                    continue

                select_num = int(cls_mask.sum() * select_ratio)

                if select_num < 1:
                    continue

                weight_c = entropy_weights[b].clone()
                weight_c[cls_mask == False] = 0

                # probability based sampling
                index_of_pseudo_labels = torch.multinomial(
                    weight_c.reshape(-1), select_num, replacement=False
                )

                pseudo_labels_mask_c = (
                    torch.zeros_like(weight_c).reshape(-1).cuda().bool()
                )
                pseudo_labels_mask_c[index_of_pseudo_labels] = True
                pseudo_labels_mask_c = pseudo_labels_mask_c.reshape(h, w)

                pseudo_labels_mask_c = (pseudo_labels_mask_c * cls_mask).bool()  # h, w

                low_entropy_mask[b, cls, :, :] = pseudo_labels_mask_c

        # combine class_mask
        low_entropy_mask = low_entropy_mask.sum(1).bool()  # b, h, w

        # replace train label with pseudo label
        pseudo_label = (pseudo_label * low_entropy_mask).long()

        # make sure we mantain ground truth
        pseudo_label[wss_mask] = train_label[wss_mask]
        new_wss_mask = pseudo_label != self.settings.ignore_cls

        return pseudo_label, new_wss_mask

    def run(self, epoch, mode="Train"):
        if mode == "Train":
            dataloader = self.train_loader
            self.model.train()
            if self.settings.distributed:
                self.train_sampler.set_epoch(epoch)
        elif mode == "Validation":
            dataloader = self.val_loader
            self.model.eval()
        else:
            raise ValueError("invalid mode: {}".format(mode))

        return_feat = (
            True
            if epoch >= self.settings.contrast_warmup and mode == "Train"
            else False
        )
        entropy_selection = (
            self.settings.entropy_selection
            if epoch >= self.settings.contrast_warmup and mode == "Train"
            else False
        )

        # init epoch-wise metrics meter
        loss_meter = pc_processor.utils.AverageMeter()
        loss_softmax_meter = pc_processor.utils.AverageMeter()
        loss_lovasz_meter = pc_processor.utils.AverageMeter()
        loss_contrast_meter = pc_processor.utils.AverageMeter()

        self.evaluator.reset()

        total_iter = len(dataloader)
        t_start = time.time()

        feature_mean = (
            torch.Tensor(self.settings.config["sensor"]["img_mean"])
            .unsqueeze(0)
            .unsqueeze(2)
            .unsqueeze(2)
            .cuda()
        )
        feature_std = (
            torch.Tensor(self.settings.config["sensor"]["img_stds"])
            .unsqueeze(0)
            .unsqueeze(2)
            .unsqueeze(2)
            .cuda()
        )

        plot_path = os.path.join(self.settings.save_path, "plot")
        os.makedirs(plot_path, exist_ok=True)

        for i, (
            input_feature,
            train_label,
            eval_label,
            _,
            tar_frame,
            seq_id,
            frame_id,
            unproj_full_labels,
            unproj_weak_labels,
            uproj_x_idx,
            uproj_y_idx,
        ) in enumerate(dataloader):

            # filter no labelled pixel appeared on train label
            if input_feature is None:
                print("!! Warning, no labelled pixels after projection")
                continue

            # record per iter
            for g in self.optimizer.param_groups:
                lr = g["lr"]
                break

            # process batch data
            # projected data
            t_process_start = time.time()
            input_feature = input_feature.cuda()  # (b, 5, h, w)
            train_label = train_label.cuda().long()  # (b, h, w)
            eval_label = eval_label.cuda().long()  # (b, h, w)
            wss_mask = train_label.gt(0)  # (b, h, w)
            eval_mask = eval_label.gt(0)  # (b, h, w)
            input_feature[:, 0:5] = (
                (input_feature[:, 0:5] - feature_mean)
                / feature_std
                * eval_mask.unsqueeze(1).expand_as(input_feature[:, 0:5])
            )  # dxyzi
            pcd_feature = input_feature

            # unprojected data
            unproj_full_labels = unproj_full_labels.cuda().long()  # (b, n)

            if self.settings.dataset == "SemanticPOSS":
                uproj_x_idx = uproj_x_idx.cuda().bool()  # (b, n)
                uproj_y_idx = uproj_y_idx.cuda().bool()  # (b, n)
            else:
                uproj_x_idx = uproj_x_idx.cuda().long()  # (b, n)
                uproj_y_idx = uproj_y_idx.cuda().long()  # (b, n)

            if mode == "Train":
                _, h, w = eval_label.shape

                # forward propergation
                output_dict = self.model(
                    pcd_feature,
                    label=train_label if return_feat else None,
                    eval_mask=wss_mask if return_feat else None,
                    return_feat=return_feat,
                )
                if "pred_2d" in output_dict.keys():
                    pred_2d = output_dict["pred_2d"]
                if "feat_2d" in output_dict.keys():
                    feat_2d = output_dict["feat_2d"]
                    _, dim_feat, h_feat, w_feat = feat_2d.shape
                    feat_2d = F.interpolate(
                        feat_2d, (h, w), mode="bilinear", align_corners=True
                    )

                # supervised loss
                total_loss = torch.tensor(0.0).cuda()

                if self.settings.loss_w_ce_2d > 0:
                    ce_loss_2d = self.criterion["focal_loss"](
                        pred_2d, train_label, mask=wss_mask
                    )
                    total_loss = total_loss + self.settings.loss_w_ce_2d * ce_loss_2d

                if self.settings.loss_w_lov_2d > 0:
                    lov_loss_2d = self.criterion["lovasz"](pred_2d, train_label)
                    total_loss = total_loss + self.settings.loss_w_lov_2d * lov_loss_2d

                # contrast learning
                if entropy_selection:
                    with torch.no_grad():
                        select_ratio = np.log(
                            1 + (1 + epoch) / self.settings.n_epochs
                        ) / np.log(
                            2
                        )  # [0, 1]
                        select_ratio = select_ratio * 0.5

                        (
                            train_label_contra,
                            wss_mask_contra,
                        ) = self.entropy_based_selection(
                            output=pred_2d,
                            wss_mask=wss_mask,
                            eval_mask=eval_mask,
                            train_label=train_label,
                            select_ratio=select_ratio,
                        )

                if self.settings.loss_w_contrast > 0 and return_feat:
                    if self.settings.distributed:
                        proto_queue = self.model.module.prototypes.detach().unsqueeze(0)
                    else:
                        proto_queue = self.model.prototypes.detach().unsqueeze(0)

                    contrast_loss = self.criterion["contrast_loss"](
                        feats=feat_2d,
                        output=pred_2d,
                        labels=train_label_contra,
                        keep_mask=wss_mask_contra,
                        proto_queue=proto_queue,
                    )

                    total_loss = (
                        total_loss + self.settings.loss_w_contrast * contrast_loss
                    )

                # backward
                total_loss = total_loss.mean()
                if self.settings.loss_w_ce_2d > 0:
                    ce_loss_2d = ce_loss_2d.mean()
                if self.settings.loss_w_lov_2d > 0:
                    lov_loss_2d = lov_loss_2d.mean()
                if self.settings.loss_w_contrast > 0 and return_feat:
                    contrast_loss = contrast_loss.mean()

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            else:
                # val mode
                with torch.no_grad():
                    output_dict = self.model(pcd_feature)
                    pred_2d = output_dict["pred_2d"]

            # measure accuracy and record loss
            with torch.no_grad():
                argmax_2d = pred_2d.argmax(dim=1)  # (b, h, w)

                # unproject 2d prediction to 3d
                for ii in range(len(train_label)):
                    if self.settings.dataset in ["SemanticKitti", "nuScenes"]:
                        unproj_argmax = argmax_2d[ii, uproj_y_idx[ii], uproj_x_idx[ii]]
                    elif self.settings.dataset == "SemanticPOSS":
                        temp_argmax = argmax_2d[ii, :].reshape(-1)  # (h, w) -> (n, )
                        temp_argmax = temp_argmax[uproj_y_idx[ii]]  # (n', ), n' != n
                        unproj_argmax = torch.zeros(40 * 1800).cuda().long()
                        unproj_argmax[: temp_argmax.shape[0]] = temp_argmax
                    else:
                        raise ValueError

                    self.evaluator.addBatch(unproj_argmax, unproj_full_labels[ii])

                mean_iou3d, class_iou3d = self.evaluator.getIoU()
                mean_acc3d, class_acc3d = self.evaluator.getAcc()
                mean_recall3d, class_recall3d = self.evaluator.getRecall()

                # sync dirtributed tensor
                if self.settings.distributed:
                    mean_acc3d = mean_acc3d.cuda()
                    mean_iou3d = mean_iou3d.cuda()
                    mean_recall3d = mean_recall3d.cuda()

                    torch.distributed.barrier()
                    torch.distributed.all_reduce(mean_acc3d)
                    torch.distributed.all_reduce(mean_iou3d)
                    torch.distributed.all_reduce(mean_recall3d)

                    mean_acc3d = mean_acc3d.cpu() / self.settings.world_size
                    mean_iou3d = mean_iou3d.cpu() / self.settings.world_size
                    mean_recall3d = mean_recall3d.cpu() / self.settings.world_size

                if mode == "Train":
                    loss_meter.update(total_loss.item(), input_feature.size(0))
                    if self.settings.loss_w_ce_2d > 0:
                        loss_softmax_meter.update(
                            ce_loss_2d.item(), input_feature.size(0)
                        )
                    if self.settings.loss_w_lov_2d > 0:
                        loss_lovasz_meter.update(
                            lov_loss_2d.item(), input_feature.size(0)
                        )
                    if self.settings.loss_w_contrast > 0 and return_feat:
                        loss_contrast_meter.update(
                            contrast_loss.item(), input_feature.size(0)
                        )

            # iteration logger
            t_process_end = time.time()

            data_cost_time = t_process_start - t_start
            process_cost_time = t_process_end - t_process_start

            self.remain_time.update(cost_time=(time.time() - t_start), mode=mode)
            remain_time = datetime.timedelta(
                seconds=self.remain_time.getRemainTime(
                    epoch=epoch, iters=i, total_iter=total_iter, mode=mode
                )
            )
            t_start = time.time()

            if self.recorder is not None:
                log_str = ">>> {} E[{:03d}|{:03d}] I[{:04d}|{:04d}] DT[{:.3f}] PT[{:.3f}] LR {:0.5f} ".format(
                    mode,
                    self.settings.n_epochs,
                    epoch + 1,
                    total_iter,
                    i + 1,
                    data_cost_time,
                    process_cost_time,
                    lr,
                )

                if mode == "Train":
                    log_str += "ALoss {:0.4f} ".format(total_loss.item())
                    if self.settings.loss_w_ce_2d > 0:
                        log_str += "CELoss {:0.4f} ".format(ce_loss_2d.item())
                    if self.settings.loss_w_lov_2d > 0:
                        log_str += "Lov {:0.4f} ".format(lov_loss_2d.item())
                    if self.settings.loss_w_contrast > 0 and return_feat:
                        log_str += "ContraLoss {:0.4f} ".format(contrast_loss.item())

                log_str += " IOU {:0.4f} ".format(mean_iou3d.item())
                log_str += "RT {}".format(remain_time)

                self.recorder.logger.info(log_str)

            if self.settings.is_debug and i > 2:
                break

        # epoch logger
        if self.recorder is not None:
            if mode == "Train":
                self.recorder.tensorboard.add_scalar(
                    tag="{}_lr".format(mode), scalar_value=lr, global_step=epoch
                )
                self.recorder.tensorboard.add_scalar(
                    tag="{}_Loss".format(mode),
                    scalar_value=loss_meter.avg,
                    global_step=epoch,
                )
                if self.settings.loss_w_ce_2d > 0:
                    self.recorder.tensorboard.add_scalar(
                        tag="{}_LossSoftmax".format(mode),
                        scalar_value=loss_softmax_meter.avg,
                        global_step=epoch,
                    )
                if self.settings.loss_w_lov_2d > 0:
                    self.recorder.tensorboard.add_scalar(
                        tag="{}_LossLovasz".format(mode),
                        scalar_value=loss_lovasz_meter.avg,
                        global_step=epoch,
                    )
                if self.settings.loss_w_contrast > 0 and return_feat:
                    self.recorder.tensorboard.add_scalar(
                        tag="{}_LossContrast".format(mode),
                        scalar_value=loss_contrast_meter.avg,
                        global_step=epoch,
                    )

            self.recorder.tensorboard.add_scalar(
                tag="{}_mean_Acc_3D".format(mode),
                scalar_value=mean_acc3d.item(),
                global_step=epoch,
            )
            self.recorder.tensorboard.add_scalar(
                tag="{}_mean_IOU_3D".format(mode),
                scalar_value=mean_iou3d.item(),
                global_step=epoch,
            )

            log_str = ">>> Epoch Finish: {} Loss {:0.4f} ".format(mode, loss_meter.avg)

            log_str += ">>> Acc {:0.4f} mIOU {:0.4F} mRecall {:0.4f}".format(
                mean_acc3d.item(), mean_iou3d.item(), mean_recall3d.item()
            )

            self.recorder.logger.info(log_str)

            # per class 3D iou
            for i, iou in enumerate(class_iou3d.cpu()):
                if i not in [0]:
                    self.recorder.logger.info(
                        "class {:02d} {} iou: {:3f}".format(
                            i, dataloader.dataset.mapped_cls_name[i], iou
                        )
                    )

            # per class 3D iou
            for i, (_, v) in enumerate(self.mapped_cls_name.items()):
                self.recorder.tensorboard.add_scalar(
                    tag="{}_IOU_{:02d}_{}".format(mode, i, v),
                    scalar_value=class_iou3d[i].item(),
                    global_step=epoch,
                )

            # log images
            output_images = self._combineTensorboradImages(
                argmax=argmax_2d[0].cpu(),
                full_label=eval_label[0].cpu(),
                weak_label=train_label[0].cpu(),
                is_weak=False,
            )
            self.recorder.tensorboard.add_image(
                "{}_Images".format(mode), output_images, epoch
            )

            if entropy_selection:
                output_images = self._combineTensorboradImages(
                    argmax=argmax_2d[0].cpu(),
                    full_label=eval_label[0].cpu(),
                    weak_label=train_label_contra[0].cpu(),
                    is_weak=False,
                )
            self.recorder.tensorboard.add_image(
                "{}_PLImages".format(mode), output_images, epoch
            )

        result_metrics = {
            "3DAcc": mean_acc3d.item(),
            "3DIOU": mean_iou3d.item(),
        }
        return result_metrics
