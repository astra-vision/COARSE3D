# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

# import re
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

# import sys

# sys.path.insert(0, "../../")
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


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output


class ResBlock(nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        dropout_rate,
        kernel_size=(3, 3),
        stride=1,
        pooling=True,
        drop_out=True,
    ):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(
            in_filters, out_filters, kernel_size=(1, 1), stride=stride
        )
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(
            out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2
        )
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(
            out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1
        )
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1, resA2, resA3), dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(nn.Module):
    def __init__(
        self, in_filters, out_filters, dropout_rate, drop_out=True, inplace=False
    ):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(
            in_filters // 4 + 2 * out_filters, out_filters, (3, 3), padding=1
        )
        self.act1 = nn.LeakyReLU(inplace=inplace)
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act2 = nn.LeakyReLU(inplace=inplace)
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2, 2), dilation=2, padding=1)
        self.act3 = nn.LeakyReLU(inplace=inplace)
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act4 = nn.LeakyReLU(inplace=inplace)
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA, skip), dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1, upE2, upE3), dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE


#
class FC(nn.Module):
    """
    a classifier, for pretrain ImageNet
    """

    def __init__(self, base_channels):
        super(FC, self).__init__()
        self.base_channels = base_channels
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_features=base_channels, out_features=1000, bias=True)

    def forward(self, x):
        x = self.pool(x)  # b, c, 1, 1
        x = x.view(-1, self.base_channels)  # b, c
        x = self.linear(x)  # b, 1000
        return x


class SEBlock(nn.Module):
    def __init__(self, inplanes, r=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.se = nn.Sequential(
            nn.Linear(inplanes, inplanes // r),
            nn.ReLU(),
            # nn.ReLU(inplace=True),
            nn.Linear(inplanes // r, inplanes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        xx = self.global_pool(x)
        xx = xx.view(xx.size(0), xx.size(1))
        se_weight = self.se(xx).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)


class SalsaNextProto(nn.Module):
    def __init__(
        self,
        in_channel=5,
        nclasses=20,
        sub_proto_size=20,
        ignore_label=0,
        use_prototype=False,
        softmax=True,
        proj_dim=256,
        projection="v1",
        classification=False,
        proto_mom=0.999,
        dataset="SemanticKitti",
    ):
        super(SalsaNextProto, self).__init__()
        self.nclasses = nclasses
        self.base_channels = 32
        self.proj_dim = proj_dim
        self.softmax = softmax
        # self.return_feat = return_feat
        # self.inplace = inplace
        self.projection = projection
        self.classification = classification
        self.use_prototype = use_prototype
        self.sub_proto_size = sub_proto_size
        self.ignore_label = ignore_label
        self.proto_mom = proto_mom
        self.dataset = dataset

        # Backbone
        self.downCntx = ResContextBlock(in_channel, self.base_channels)
        self.downCntx2 = ResContextBlock(self.base_channels, self.base_channels)
        self.downCntx3 = ResContextBlock(self.base_channels, self.base_channels)

        self.resBlock1 = ResBlock(
            self.base_channels,
            2 * self.base_channels,
            0.2,
            pooling=True,
            drop_out=False,
        )
        self.resBlock2 = ResBlock(
            2 * self.base_channels, 2 * 2 * self.base_channels, 0.2, pooling=True
        )
        self.resBlock3 = ResBlock(
            2 * 2 * self.base_channels, 2 * 4 * self.base_channels, 0.2, pooling=True
        )
        self.resBlock4 = ResBlock(
            2 * 4 * self.base_channels, 2 * 4 * self.base_channels, 0.2, pooling=True
        )
        self.resBlock5 = ResBlock(
            2 * 4 * self.base_channels, 2 * 4 * self.base_channels, 0.2, pooling=False
        )

        if self.classification:
            self.fc = FC(2 * 4 * self.base_channels)

        self.upBlock1 = UpBlock(2 * 4 * self.base_channels, 4 * self.base_channels, 0.2)
        self.upBlock2 = UpBlock(4 * self.base_channels, 4 * self.base_channels, 0.2)
        self.upBlock3 = UpBlock(4 * self.base_channels, 2 * self.base_channels, 0.2)
        self.upBlock4 = UpBlock(
            2 * self.base_channels, self.base_channels, 0.2, drop_out=False
        )  # , inplace=self.inplace)

        self.cls_head = nn.Conv2d(self.base_channels, nclasses, kernel_size=(1, 1))

        self.projector = ProjectionV1(self.base_channels * 22, proj_dim)

        self.prototypes = nn.Parameter(
            torch.randn(nclasses, sub_proto_size, proj_dim), requires_grad=False
        )
        trunc_normal_(self.prototypes, std=0.02)

        self.feat_norm = nn.LayerNorm(proj_dim)
        self.mask_norm = nn.LayerNorm(nclasses)

    def _make_pred_layer(
        self, block, inplanes, dilation_series, padding_series, num_classes, channel
    ):
        return block(
            inplanes, dilation_series, padding_series, num_classes, channel=channel
        )

    def prototype_learning(
        self, out_feat, nearest_proto_distance, label, eval_mask, feat_proto_sim
    ):
        pred_seg = torch.max(nearest_proto_distance, 1)[1]  # the idx of the nearest
        mask = label == pred_seg.view(-1)

        cosine_similarity = feat_proto_sim.reshape(feat_proto_sim.shape[0], -1)

        proto_logits = cosine_similarity
        proto_target = torch.zeros_like(label).float()

        # clustering for each class, on line
        protos = self.prototypes.data.clone()
        for id_c in range(self.nclasses):
            if id_c == self.ignore_label:
                continue

            init_q = feat_proto_sim[..., id_c]  # n, k, cls => n, k
            init_q = init_q[label == id_c, ...]  #
            if init_q.shape[0] == 0:  # no such class
                continue

            q, indexs = distributed_sinkhorn(init_q)
            # q: one-hot prototype label
            # indexes: prototype label

            m_c = mask[label == id_c]  # [n]

            feat_c = out_feat[label == id_c, ...]  # [n, 720]

            m_c_tile = repeat(m_c, "n -> n tile", tile=self.sub_proto_size)  # [n, p]

            m_q = q * m_c_tile  # (n x p) (n x p) -> (n x p) masked one hot label

            m_c_tile = repeat(m_c, "n -> n tile", tile=feat_c.shape[-1])  # [n, dim]

            c_q = feat_c * m_c_tile  # [n, dim]

            f = m_q.transpose(0, 1) @ c_q  # (p, n) (n, dim) => (p, dim)

            n = torch.sum(m_q, dim=0)  # (n, p) => p

            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1)  # [p, dim]

                new_value = momentum_update(
                    old_value=protos[id_c, n != 0, :],
                    new_value=f[n != 0, :],
                    momentum=self.proto_mom,
                    debug=False,
                )
                protos[id_c, n != 0, :] = new_value

            proto_target[label == id_c] = indexs.float() + (
                self.sub_proto_size * id_c
            )  # (n, ) cls*k classes totally

        self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)

        # syn prototypes on gpus
        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_logits, proto_target

    def forward(
        self,
        x,
        label=None,
        eval_mask=None,
        return_feat=True,
        proto_loss=False,
        proto_pl=None,
    ):

        bs = 1
        c = 5
        h = 64
        w = 2048
        x = torch.randn((bs, c, h, w)).cuda()
        label = torch.ones((bs, h, w)) * 2
        label = label.cuda()
        eval_mask = torch.ones((bs, h, w)).cuda()

        b, c, h, w = x.shape

        # pad
        if self.dataset == "SemanticPOSS":
            new_x = torch.zeros((b, c, h + 8, w + 8)).cuda()
            new_x[:, :, :h, :w] = x
            x = new_x
            b, c, h, w = x.shape

        assert h % 16 == 0, w % 16 == 0

        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)

        # if use ImageNet dataset to pretrain the encoder
        if self.classification:
            cls_out = self.fc(down5c)
            return cls_out

        up4e = self.upBlock1(down5c, down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)

        logits = self.cls_head(up1e)

        # de pad
        if self.dataset == "SemanticPOSS":
            logits = logits[:, :, :-8, :-8]

        probs = F.softmax(logits, dim=1)

        return_dict = {}
        return_dict["pred_2d"] = probs

        # contrast learning
        if return_feat:
            _, _, h, w = logits.shape
            h = int(h / 2)
            w = int(w / 2)
            feat0 = F.interpolate(
                down0b, size=(h, w), mode="bilinear", align_corners=True
            )
            feat1 = F.interpolate(
                down1b, size=(h, w), mode="bilinear", align_corners=True
            )
            feat2 = F.interpolate(
                down2b, size=(h, w), mode="bilinear", align_corners=True
            )
            feat3 = F.interpolate(
                down3b, size=(h, w), mode="bilinear", align_corners=True
            )
            feat = torch.cat([feat0, feat1, feat2, feat3], 1)
            embedding = self.projector(feat)

            embedding = F.normalize(embedding, p=2, dim=1)  # b, dim, h, w

            _, _, h_lbl, w_lbl = logits.shape
            embedding = F.interpolate(
                embedding, (h_lbl, w_lbl), mode="bilinear", align_corners=True
            )

            return_dict["feat_2d"] = embedding  # b, dim, h, w

            if self.use_prototype and label is not None and eval_mask is not None:
                # prototype learning
                b, dim, h, w = embedding.shape
                out_feat = rearrange(embedding, "b c h w -> (b h w) c")
                out_feat = self.feat_norm(out_feat)  # (n, dim)

                # cosine sim
                out_feat = l2_normalize(out_feat)
                self.prototypes.data.copy_(l2_normalize(self.prototypes))

                feat_proto_sim = torch.einsum("nd,kmd->nmk", out_feat, self.prototypes)

                nearest_proto_distance = torch.amax(feat_proto_sim, dim=1)
                nearest_proto_distance = self.mask_norm(nearest_proto_distance)
                nearest_proto_distance = rearrange(
                    nearest_proto_distance, "(b h w) k -> b k h w", b=b, h=h
                )

                label_expand = label.view(-1)
                eval_mask_expand = eval_mask.view(-1)

                if proto_pl is not None:
                    self.prototypes = nn.Parameter(
                        proto_pl.clone(), requires_grad=False
                    )

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
