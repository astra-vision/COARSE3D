import torch
import torch.nn as nn
import torch.nn.functional as F

torch.random.manual_seed(0)


class ContrastMEMLoss(nn.Module):
    def __init__(self,
                 ignore_label=0,
                 temperature=0.1,
                 base_temperature=0.07,
                 num_anchor=50,
                 is_debug=False,
                 ):
        super(ContrastMEMLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_anchor = num_anchor
        self.ignore_label = ignore_label

        self.is_debug = is_debug
        self.sub_proto = True

    def forward(self,
                feats=None,
                output=None,
                labels=None,
                keep_mask=None,
                proto_queue=None,
                ):

        labels = labels.clone()
        if keep_mask is not None:
            labels[keep_mask.bool() == False] = self.ignore_label

        assert proto_queue is not None
        proto_queue = proto_queue.squeeze(0)

        if self.is_debug:
            print('queue size, max views : ', proto_queue.shape)

        if output is not None:
            entropy = -torch.sum(output * torch.log(output + 1e-10), dim=1)  # b, h, w
            entropy = entropy * entropy
            entropy_weights = torch.exp(-1 * entropy)
        else:
            entropy = None
            entropy_weights = None

        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size, dim, h, w = feats.shape
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(batch_size, -1, dim)  # b, n
        labels = labels.contiguous().view(batch_size, -1)  # b, n

        if entropy is not None:
            entropy_weights = entropy_weights.contiguous().view(batch_size, -1)  # b, n
        else:
            entropy_weights = None

        # entropy based anchor sampling
        feats_, labels_ = self.anchor_sampling(feats, labels, weights=entropy_weights)
        assert len(feats) > 0, 'no anchor feature is selected for loss'

        # compute loss based on prototype queue
        loss = self._contrastive(feats_, labels_, queue=proto_queue)

        return loss

    def anchor_sampling(self, X, y_hat, weights=None):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y).long()

            this_classes = [x for x in this_classes if x != self.ignore_label]

            classes.append(this_classes)
            total_classes = total_classes + len(this_classes)

        if total_classes == 0:
            return None, None

        X_ = torch.zeros((total_classes, self.num_anchor, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]  # label
            this_classes = classes[ii]  # classes in the current batch

            for cls_id in this_classes:
                if cls_id == self.ignore_label:
                    continue

                if (this_y_hat == cls_id).sum() < 1:
                    continue

                assert weights is not None
                weight_c = weights[ii].clone()
                weight_c[this_y_hat != cls_id] = 0
                keep_indices = torch.multinomial(weight_c.reshape(-1), self.num_anchor, replacement=True)

                X_[X_ptr, :keep_indices.shape[0], :] = X[ii, keep_indices, :].squeeze(1)

                y_[X_ptr] = cls_id
                X_ptr = X_ptr + 1

        if X_ptr < y_.shape[0]:
            X_ = X_[:X_ptr]
            y_ = y_[:X_ptr]

        return X_, y_

    def _expand_queue(self, queue):
        class_num, cache_size, feat_size = queue.shape

        X_ = torch.zeros(((class_num - 1) * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros(((class_num - 1) * cache_size, 1)).float().cuda()

        sample_ptr = 0
        for ii in range(class_num):
            if ii == 0:
                continue

            perm = torch.randperm(int(cache_size))
            this_q = queue[ii, perm[:cache_size], :]

            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        cls_num, num_anchor = X_anchor.shape[0], X_anchor.shape[1]

        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        X_contrast, y_contrast = self._expand_queue(queue)
        y_contrast = y_contrast.contiguous().view(-1, 1)
        contrast_count = 1
        contrast_feature = X_contrast

        # positive mask
        mask = torch.eq(y_anchor, y_contrast.T).float().cuda()

        # cosine similarity
        anchor_feature = F.normalize(anchor_feature, p=2, dim=-1)  # (n, dim)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=-1)
        anchor_dot_contrast = torch.einsum('nd,kd->nk', anchor_feature,
                                           contrast_feature)  # (n, dim) (cls, dim) => (n, cls)

        anchor_dot_contrast = torch.div(anchor_dot_contrast, self.temperature)

        # remove logits_max for numerical stability. not necessary.
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(num_anchor, contrast_count)
        neg_mask = 1 - mask

        # the sum of logits of all negative pairs of one anchor
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        # logit of each sample pair
        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits + 1e-6)

        # keep positive logits, divide positives number
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
