#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

import sys

sys.path.insert(0, "../../")
from pc_processor.models.projector import ProjectionV1
from pc_processor.models.sinkhorn import distributed_sinkhorn


def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print(
            "# old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
                momentum,
                torch.norm(old_value, p=2),
                (1 - momentum),
                torch.norm(new_value, p=2),
                torch.norm(update, p=2),
            )
        )
    return update


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


class SqueezeSegV3Proto(nn.Module):
    def __init__(
        self,
        nclasses,
        dataset="SemanticKitti",
        path=None,
        path_append="",
        strict=False,
        layers=21,
        # added params
        proj_dim=256,
        projection="v1",
        proj_feat="mix",
        l2_norm=True,
        proto_mom=0.999,
        ignore_label=0,
        sub_proto_size=20,
        use_prototype=False,
        pred_3d=False,
    ):
        super().__init__()
        self.nclasses = nclasses
        self.path = path
        self.path_append = path_append
        self.dataset = dataset
        self.strict = False

        # proto params
        self.l2_norm = l2_norm
        self.use_prototype = use_prototype
        self.sub_proto_size = sub_proto_size
        self.ignore_label = ignore_label
        self.proto_mom = proto_mom
        self.projection = projection
        self.proj_feat = proj_feat

        self.backbone = Backbone(layers=layers)

        self.decoder = Decoder(OS=8, feature_depth=self.backbone.get_last_depth())

        self.head1 = nn.Sequential(
            nn.Dropout2d(p=0.01),
            nn.Conv2d(256, self.nclasses, kernel_size=1, stride=1, padding=0),
        )

        self.head2 = nn.Sequential(
            nn.Dropout2d(p=0.01),
            nn.Conv2d(256, self.nclasses, kernel_size=1, stride=1, padding=0),
        )

        self.head3 = nn.Sequential(
            nn.Dropout2d(p=0.01),
            nn.Conv2d(128, self.nclasses, kernel_size=1, stride=1, padding=0),
        )

        self.head4 = nn.Sequential(
            nn.Dropout2d(p=0.01),
            nn.Conv2d(64, self.nclasses, kernel_size=1, stride=1, padding=0),
        )
        self.head5 = nn.Sequential(
            nn.Dropout2d(p=0.01),
            nn.Conv2d(32, self.nclasses, kernel_size=3, stride=1, padding=1),
        )

        # print number of parameters and the ones requiring gradients
        weights_total = sum(p.numel() for p in self.parameters())
        weights_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total number of parameters: ", weights_total)
        print("Total number of parameters requires_grad: ", weights_grad)

        # breakdown by layer
        weights_enc = sum(p.numel() for p in self.backbone.parameters())
        weights_dec = sum(p.numel() for p in self.decoder.parameters())
        weights_head = (
            sum(p.numel() for p in self.head1.parameters())
            + sum(p.numel() for p in self.head2.parameters())
            + sum(p.numel() for p in self.head3.parameters())
            + sum(p.numel() for p in self.head4.parameters())
            + sum(p.numel() for p in self.head5.parameters())
        )
        print("Param encoder ", weights_enc)
        print("Param decoder ", weights_dec)
        print("Param head ", weights_head)

        # get weights
        if path is not None:
            # try backbone
            try:
                w_dict = torch.load(
                    path + "/backbone", map_location=lambda storage, loc: storage
                )
                self.backbone.load_state_dict(w_dict, strict=True)
                print("Successfully loaded model backbone weights")
            except Exception as e:
                print()
                print("Couldn't load backbone, using random weights. Error: ", e)
                if strict:
                    print(
                        "I'm in strict mode and failure to load weights blows me up :)"
                    )
                    raise e

            # try decoder
            try:
                w_dict = torch.load(
                    path + "/segmentation_decoder",
                    map_location=lambda storage, loc: storage,
                )
                self.decoder.load_state_dict(w_dict, strict=True)
                print("Successfully loaded model decoder weights")
            except Exception as e:
                print("Couldn't load decoder, using random weights. Error: ", e)
                if strict:
                    print(
                        "I'm in strict mode and failure to load weights blows me up :)"
                    )
                    raise e

            # try head
            try:
                print(path_append + "./segmentation_head1")
                w_dict = torch.load(
                    path + "/segmentation_head1",
                    map_location=lambda storage, loc: storage,
                )
                self.head1.load_state_dict(w_dict, strict=True)
                print("Successfully loaded model head weights")
            except Exception as e:
                print("Couldn't load head, using random weights. Error: ", e)
                if strict:
                    print(
                        "I'm in strict mode and failure to load weights blows me up :)"
                    )
                    raise e
            try:
                w_dict = torch.load(
                    path + "/segmentation_head2",
                    map_location=lambda storage, loc: storage,
                )
                self.head2.load_state_dict(w_dict, strict=True)
                print("Successfully loaded model head weights")
            except Exception as e:
                print("Couldn't load head, using random weights. Error: ", e)
                if strict:
                    print(
                        "I'm in strict mode and failure to load weights blows me up :)"
                    )
                    raise e
            try:
                w_dict = torch.load(
                    path + "/segmentation_head3",
                    map_location=lambda storage, loc: storage,
                )
                self.head3.load_state_dict(w_dict, strict=True)
                print("Successfully loaded model head weights")
            except Exception as e:
                print("Couldn't load head, using random weights. Error: ", e)
                if strict:
                    print(
                        "I'm in strict mode and failure to load weights blows me up :)"
                    )
                    raise e

            try:
                w_dict = torch.load(
                    path + "/segmentation_head4",
                    map_location=lambda storage, loc: storage,
                )
                self.head4.load_state_dict(w_dict, strict=True)
                print("Successfully loaded model head weights")
            except Exception as e:
                print("Couldn't load head, using random weights. Error: ", e)
                if strict:
                    print(
                        "I'm in strict mode and failure to load weights blows me up :)"
                    )
                    raise e

            try:
                w_dict = torch.load(
                    path + "/segmentation_head5",
                    map_location=lambda storage, loc: storage,
                )
                self.head5.load_state_dict(w_dict, strict=True)
                print("Successfully loaded model head weights")
            except Exception as e:
                print("Couldn't load head, using random weights. Error: ", e)
                if strict:
                    print(
                        "I'm in strict mode and failure to load weights blows me up :)"
                    )
                    raise e
        else:
            print("No path to pretrained, using random init.")

        # contrast define
        if self.projection == "v1":
            self.projector = ProjectionV1(480, proj_dim)  # mix
        else:
            raise NotImplementedError

        # memory bank
        self.prototypes = nn.Parameter(
            torch.randn(nclasses, sub_proto_size, proj_dim),
            # requires_grad=True,
            requires_grad=False,
        )
        trunc_normal_(self.prototypes, std=0.02)

        self.feat_norm = nn.LayerNorm(proj_dim)
        self.mask_norm = nn.LayerNorm(nclasses)

        return

    def prototype_learning(
        self,
        out_feat,
        nearest_proto_distance,
        label,
        eval_mask,
        feat_proto_sim,
        cosine=False,
        cosine_abs=True,
        euclidean=False,
        l1_dist=False,
        kl_dist=False,
        weighted_sum=False,
    ):
        """
        :param out_feat: [h*w, dim] pixel feature
        :param nearest_proto_distance: [bs, cls_num, h, w] 
        :param label: [h*w] segmentation label
        :param feat_proto_sim: [h*w, sub_cluster, cls_num]
        """
        pred_seg = torch.max(nearest_proto_distance, 1)[
            1
        ]  # [1, 128, 256] the idx of the nearest
        mask = label == pred_seg.view(-1)  
        # mask = eval_mask.view(-1)
        # (b*h*w, dim) (cls*k, dim) => (b*h*w, cls*k) # from cls classes to cls*k classes
        # cosine_similarity = torch.mm(out_feat, self.prototypes.view(-1, self.prototypes.shape[-1]).t())  
        cosine_similarity = feat_proto_sim.reshape(feat_proto_sim.shape[0], -1)

        proto_logits = cosine_similarity
        # proto_target = label.clone().float()  # (n, )
        proto_target = torch.zeros_like(label).float()  # (n, )

        # clustering for each class, on line
        protos = self.prototypes.data.clone()
        for id_c in range(self.nclasses):
            if id_c == self.ignore_label:
                continue

            init_q = feat_proto_sim[
                ..., id_c
            ]  # n, k, cls => n, k 
            init_q = init_q[label == id_c, ...]  #
            if init_q.shape[0] == 0:  # no such class
                continue

            q, indexs = distributed_sinkhorn(init_q)
            # q: (n, 10) one-hot prototype label
            # indexes: (n, ) prototype label # torch.unique(indexs) = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            m_c = mask[label == id_c]  # [n]

            feat_c = out_feat[label == id_c, ...]  # [n, dim]

            m_c_tile = repeat(m_c, "n -> n tile", tile=self.sub_proto_size)  # [n, p]

            m_q = q * m_c_tile  # (n x p) (n x p) -> (n x p) masked one hot label

            m_c_tile = repeat(m_c, "n -> n tile", tile=feat_c.shape[-1])  # [n, dim]

            c_q = feat_c * m_c_tile  # [n, dim]

            if weighted_sum:
                weight = 1 - init_q  # 1 - (n, p) => (n, p)
                f = weight.transpose(0, 1) @ c_q  # (p, n) (n, dim) => (p, dim)
            else:
                f = m_q.transpose(0, 1) @ c_q  # (p, n) (n, dim) => (p, dim)

            n = torch.sum(m_q, dim=0)  # (n, p) => p

            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1)  # [p, 720]

                new_value = momentum_update(
                    old_value=protos[id_c, n != 0, :],
                    new_value=f[n != 0, :],
                    momentum=self.proto_mom,
                    # debug=True if id_c == 1 else False,
                    debug=False,
                )  # [p, dim]
                protos[id_c, n != 0, :] = new_value  # [cls, p, dim]

            proto_target[label == id_c] = indexs.float() + (
                self.sub_proto_size * id_c
            )  # (n, ) cls*k classes totally

        self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)
        # syn prototypes on gpus
        if dist.is_available() and dist.is_initialized():
            protos = (
                self.prototypes.data.clone()
            )  # [class_num, sub_cls, feat_dim]
            dist.all_reduce(
                protos.div_(dist.get_world_size())
            )  # default of all_reduce is sum
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        # if we don't use proto loss, no need to return  proto_logits, proto_target
        return proto_logits, proto_target

    def forward(
        self,
        x,
        label=None,
        eval_mask=None,
        return_feat=True,
        proto_loss=False,
        proto_pl=None,
        unproj_data=None,
    ):
        
        # b, c, h, w = x.shape

        # # pad
        # if self.dataset == 'SemanticPOSS':
        #     new_x = torch.zeros((b, c, h + 8, w + 8)).cuda()
        #     new_x[:, :, :h, :w] = x
        #     x = new_x
        #     b, c, h, w = x.shape

        feature, skips = self.backbone(x)

        y = self.decoder(feature, skips)

        z1 = self.head5(y[0])
        z1 = F.softmax(z1, dim=1)

        # if self.dataset == 'SemanticPOSS':
        #     z1 = z1[:, :, :-8, :-8]

        # z2 = self.head4(y[1])
        # z2 = F.softmax(z2, dim=1)
        #
        # z3 = self.head3(y[2])
        # z3 = F.softmax(z3, dim=1)
        #
        # z4 = self.head2(y[3])
        # z4 = F.softmax(z4, dim=1)
        #
        # z5 = self.head1(y[4])
        # z5 = F.softmax(z5, dim=1)
        return_dict = {}
        return_dict["pred_2d"] = z1.contiguous()

        # -----------------------------------------------------
        # contrast learning
        # -----------------------------------------------------

        if return_feat:
            _, _, h, w = z1.shape
            h = int(h / 2)
            w = int(w / 2)
            feat0 = F.interpolate(
                skips[1], size=(h, w), mode="bilinear", align_corners=True
            )
            feat1 = F.interpolate(
                skips[2], size=(h, w), mode="bilinear", align_corners=True
            )
            feat2 = F.interpolate(
                skips[4], size=(h, w), mode="bilinear", align_corners=True
            )
            feat3 = F.interpolate(
                feature, size=(h, w), mode="bilinear", align_corners=True
            )
            feat = torch.cat([feat0, feat1, feat2, feat3], 1)  # 480
            embedding = self.projector(feat)

            embedding = F.normalize(embedding, p=2, dim=1)  # b, dim, h, w
            _, _, h_lbl, w_lbl = z1.shape
            embedding = F.interpolate(
                embedding, (h_lbl, w_lbl), mode="bilinear", align_corners=True
            )
            return_dict["feat_2d"] = embedding  # b, dim, h, w
        else:
            embedding = None

        if self.use_prototype and label is not None and eval_mask is not None:
            # prototype learning
            b, dim, h, w = embedding.shape
            out_feat = rearrange(embedding, "b c h w -> (b h w) c")
            out_feat = self.feat_norm(out_feat)  # (n, dim)

            # cosine sim
            out_feat = l2_normalize(out_feat)  # cosine sim norm
            self.prototypes.data.copy_(l2_normalize(self.prototypes))

            feat_proto_sim = torch.einsum(
                "nd,kmd->nmk", out_feat, self.prototypes
            )  # [n, dim], [csl, p, dim] -> [n, p, cls]

            nearest_proto_distance = torch.amax(feat_proto_sim, dim=1)
            nearest_proto_distance = self.mask_norm(nearest_proto_distance)
            nearest_proto_distance = rearrange(
                nearest_proto_distance, "(b h w) k -> b k h w", b=b, h=h
            )

            label_expand = label.view(-1)
            eval_mask_expand = eval_mask.view(-1)

            if proto_pl is not None:
                self.prototypes = nn.Parameter(proto_pl.clone(), requires_grad=False)

            if proto_loss:
                contrast_logits, contrast_target = self.prototype_learning(
                    out_feat,
                    nearest_proto_distance,
                    label_expand,
                    eval_mask_expand,
                    feat_proto_sim,
                )

                return_dict["contrast_logits"] = contrast_logits
                return_dict["contrast_target"] = contrast_target
        return return_dict


class SACBlock(nn.Module):
    def __init__(self, inplanes, expand1x1_planes, bn_d=0.1):
        super(SACBlock, self).__init__()
        self.inplanes = inplanes
        self.bn_d = bn_d

        self.attention_x = nn.Sequential(
            nn.Conv2d(3, 9 * self.inplanes, kernel_size=7, padding=3),
            nn.BatchNorm2d(9 * self.inplanes, momentum=0.1),
        )

        self.position_mlp_2 = nn.Sequential(
            nn.Conv2d(9 * self.inplanes, self.inplanes, kernel_size=1),
            nn.BatchNorm2d(self.inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        xyz = input[0]
        new_xyz = input[1]
        feature = input[2]
        N, C, H, W = feature.size()

        new_feature = F.unfold(feature, kernel_size=3, padding=1).view(N, -1, H, W)
        attention = F.sigmoid(self.attention_x(new_xyz))
        new_feature = new_feature * attention
        new_feature = self.position_mlp_2(new_feature)
        fuse_feature = new_feature + feature

        return xyz, new_xyz, fuse_feature


# ******************************************************************************

# number of layers per model
model_blocks = {
    21: [1, 1, 2, 2, 1],
    53: [1, 2, 8, 8, 4],
}


class Backbone(nn.Module):
    """
    Class for DarknetSeg. Subclasses PyTorch's own "nn" module
    """

    def __init__(self, layers):
        super(Backbone, self).__init__()
        self.use_range = True
        self.use_remission = True
        self.drop_prob = 0.01
        self.bn_d = 0.01
        self.OS = 8
        self.layers = layers
        self.use_xyz = True

        print("!! Squeezesegv3 " + str(self.layers) + " Backbone")
        self.input_depth = 0
        self.input_idxs = []
        if self.use_range:
            self.input_depth += 1
            self.input_idxs.append(0)
        if self.use_xyz:
            self.input_depth += 3
            self.input_idxs.extend([1, 2, 3])
        if self.use_remission:
            self.input_depth += 1
            self.input_idxs.append(4)
        print("Depth of backbone input = ", self.input_depth)

        self.strides = [2, 2, 2, 1, 1]

        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Original OS: ", current_os)

        if self.OS > current_os:
            print(
                "Can't do OS, ",
                self.OS,
                " because it is bigger than original ",
                current_os,
            )
        else:

            for i, stride in enumerate(reversed(self.strides), 0):
                if int(current_os) != self.OS:
                    if stride == 2:
                        current_os /= 2
                        self.strides[-1 - i] = 1
                    if int(current_os) == self.OS:
                        break
            print("New OS: ", int(current_os))
            print("Strides: ", self.strides)

        assert self.layers in model_blocks.keys()

        self.blocks = model_blocks[self.layers]

        self.conv1 = nn.Conv2d(
            self.input_depth, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
        self.relu1 = nn.LeakyReLU(0.1)

        self.enc1 = self._make_enc_layer(
            SACBlock,
            [32, 64],
            self.blocks[0],
            stride=self.strides[0],
            DS=True,
            bn_d=self.bn_d,
        )
        self.enc2 = self._make_enc_layer(
            SACBlock,
            [64, 128],
            self.blocks[1],
            stride=self.strides[1],
            DS=True,
            bn_d=self.bn_d,
        )
        self.enc3 = self._make_enc_layer(
            SACBlock,
            [128, 256],
            self.blocks[2],
            stride=self.strides[2],
            DS=True,
            bn_d=self.bn_d,
        )
        self.enc4 = self._make_enc_layer(
            SACBlock,
            [256, 256],
            self.blocks[3],
            stride=self.strides[3],
            DS=False,
            bn_d=self.bn_d,
        )
        self.enc5 = self._make_enc_layer(
            SACBlock,
            [256, 256],
            self.blocks[4],
            stride=self.strides[4],
            DS=False,
            bn_d=self.bn_d,
        )

        self.dropout = nn.Dropout2d(self.drop_prob)

        self.last_channels = 256

    def _make_enc_layer(self, block, planes, blocks, stride, DS, bn_d=0.1):
        layers = []

        inplanes = planes[0]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), block(inplanes, planes, bn_d)))
        if DS == True:
            layers.append(
                (
                    "conv",
                    nn.Conv2d(
                        planes[0],
                        planes[1],
                        kernel_size=3,
                        stride=[1, stride],
                        dilation=1,
                        padding=1,
                        bias=False,
                    ),
                )
            )
            layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
            layers.append(("relu", nn.LeakyReLU(0.1)))

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, xyz, feature, layer, skips, os, flag=True):
        new_xyz = xyz
        if flag == True:
            xyz, new_xyz, y = layer[:-3]([xyz, new_xyz, feature])
            y = layer[-3:](y)
            xyz = F.upsample_bilinear(xyz, size=[xyz.size()[2], xyz.size()[3] // 2])
        else:
            xyz, new_xyz, y = layer([xyz, new_xyz, feature])
        if y.shape[2] < feature.shape[2] or y.shape[3] < feature.shape[3]:
            skips[os] = feature.detach()
            os *= 2
        feature = self.dropout(y)
        return xyz, feature, skips, os

    def forward(self, feature):
        skips = {}
        os = 1
        xyz = feature[:, 1:4, :, :]
        feature = self.relu1(self.bn1(self.conv1(feature)))

        xyz, feature, skips, os = self.run_layer(xyz, feature, self.enc1, skips, os)
        xyz, feature, skips, os = self.run_layer(xyz, feature, self.enc2, skips, os)
        xyz, feature, skips, os = self.run_layer(xyz, feature, self.enc3, skips, os)
        xyz, feature, skips, os = self.run_layer(
            xyz, feature, self.enc4, skips, os, flag=False
        )
        xyz, feature, skips, os = self.run_layer(
            xyz, feature, self.enc5, skips, os, flag=False
        )

        return feature, skips

    def get_last_depth(self):
        return self.last_channels

    def get_input_depth(self):
        return self.input_depth


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, bn_d=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(
            planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


# ******************************************************************************


class Decoder(nn.Module):
    """
    Class for DarknetSeg. Subclasses PyTorch's own "nn" module
    """

    def __init__(self, OS=32, feature_depth=1024):
        super(Decoder, self).__init__()
        self.backbone_OS = OS
        self.backbone_feature_depth = feature_depth
        self.drop_prob = 0.01
        self.bn_d = 0.01

        # stride play
        self.strides = [1, 1, 2, 2, 2]
        # check current stride
        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Decoder original OS: ", int(current_os))
        # redo strides according to needed stride
        for i, stride in enumerate(self.strides):
            if int(current_os) != self.backbone_OS:
                if stride == 2:
                    current_os /= 2
                    self.strides[i] = 1
                if int(current_os) == self.backbone_OS:
                    break
        print("Decoder new OS: ", int(current_os))
        print("Decoder strides: ", self.strides)

        # decoder
        self.dec5 = self._make_dec_layer(
            BasicBlock,
            [self.backbone_feature_depth, 256],
            bn_d=self.bn_d,
            stride=self.strides[0],
        )
        self.dec4 = self._make_dec_layer(
            BasicBlock, [256, 256], bn_d=self.bn_d, stride=self.strides[1]
        )
        self.dec3 = self._make_dec_layer(
            BasicBlock, [256, 128], bn_d=self.bn_d, stride=self.strides[2]
        )
        self.dec2 = self._make_dec_layer(
            BasicBlock, [128, 64], bn_d=self.bn_d, stride=self.strides[3]
        )
        self.dec1 = self._make_dec_layer(
            BasicBlock, [64, 32], bn_d=self.bn_d, stride=self.strides[4]
        )

        # layer list to execute with skips
        self.layers = [self.dec5, self.dec4, self.dec3, self.dec2, self.dec1]

        # for a bit of fun
        self.dropout = nn.Dropout2d(self.drop_prob)

        # last channels
        self.last_channels = 32

    def _make_dec_layer(self, block, planes, bn_d=0.1, stride=2, flag=True):
        layers = []

        #  downsample
        if stride == 2:
            layers.append(
                (
                    "upconv",
                    nn.ConvTranspose2d(
                        planes[0],
                        planes[1],
                        kernel_size=[1, 4],
                        stride=[1, 2],
                        padding=[0, 1],
                    ),
                )
            )
        else:
            layers.append(
                ("conv", nn.Conv2d(planes[0], planes[1], kernel_size=3, padding=1))
            )
        layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
        layers.append(("relu", nn.LeakyReLU(0.1)))

        #  blocks
        layers.append(("residual", block(planes[1], planes, bn_d)))

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer, skips, os):
        feats = layer(x)  # up
        if feats.shape[-1] > x.shape[-1]:
            os //= 2  # match skip
            feats = feats + skips[os].detach()  # add skip
        x = feats
        return x, skips, os

    def forward(self, x, skips):

        os = self.backbone_OS

        # run layers
        x1, skips, os = self.run_layer(x, self.dec5, skips, os)
        x2, skips, os = self.run_layer(x1, self.dec4, skips, os)
        x3, skips, os = self.run_layer(x2, self.dec3, skips, os)
        x4, skips, os = self.run_layer(x3, self.dec2, skips, os)
        x5, skips, os = self.run_layer(x4, self.dec1, skips, os)

        x5 = self.dropout(x5)

        return [x5, x4, x3, x2, x1]

    def get_last_depth(self):
        return self.last_channels


if __name__ == "__main__":

    def test_model():
        bs = 1
        c = 5
        h = 40  
        w = 1800  
        x = torch.randn((bs, c, h, w)).cuda()
        label = torch.ones_like(x).cuda()
        eval_mask = torch.ones_like(x).cuda()
        model = SqueezeSegV3Proto(
            nclasses=20,
            proj_dim=128,
            projection="v1",
            proj_feat="mix",
            l2_norm=True,
            proto_mom=0.999,
            ignore_label=0,
            sub_proto_size=20,
            use_prototype=True,
            # dataset='SemanticPOSS',
        ).cuda()
        z1 = model(
            x,
            label=label,
            eval_mask=eval_mask,
            return_feat=True,
            proto_loss=False,
            proto_pl=None,
        )
        print(z1["feat_2d"].shape)  # b, c, h, w
        print(z1["pred_2d"].shape)  # b, c, h, w
        print(model.prototypes.shape)  # cls, proto, dim
        