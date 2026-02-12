"""
Loss functions for few-shot learning
Original implementation: Takumi Kobayashi
Refactoring and extensions: Tong Wu
"""

import torch
import torch.distributed.nn
from torch import distributed, nn

from src import utils

INF = 1e9


def masked_logit(D, M):
    return D.masked_fill(~M, -INF)


def l2_dist(xq, xs):
    return -torch.pow(torch.cdist(xq, xs), 2).div(2)


def l1_dist(xq, xs):
    return -torch.cdist(xq, xs, p=1)


logit_funcs = {"l2_dist": l2_dist, "l1_dist": l1_dist}


class NCALoss(nn.Module):
    def __init__(self, T=0.9, logit="l2_dist", **kwargs):
        super().__init__()
        self.logit_func = logit_funcs[logit]
        self.xe = nn.CrossEntropyLoss()
        self.T = T

    def forward(self, xq, yq, xs, ys, pos):
        with torch.no_grad():
            class_mask = yq.view(-1, 1) == ys.view(1, -1)
            idx = (class_mask.sum(-1) > 1).cpu()
            pos = torch.tensor(pos, device=yq.device)
            ind = torch.arange(len(pos), device=yq.device)
            yq_new = torch.zeros_like(yq)

        logit = self.logit_func(xq, xs) / self.T
        logit[ind, pos] *= 0
        logit[ind[idx], pos[idx]] -= INF
        pos_logit = torch.logsumexp(masked_logit(logit, class_mask), 1, keepdim=True)
        neg_logit = torch.logsumexp(masked_logit(logit, ~class_mask), 1, keepdim=True)

        return self.xe(torch.cat([pos_logit, neg_logit], 1), yq_new)


class ProtoNetLoss(nn.Module):
    def __init__(self, T=1.0, logit="l2_dist", **kwargs):
        super().__init__()
        self.logit_func = logit_funcs[logit]
        self.xe = nn.CrossEntropyLoss()
        self.T = T

    def forward(self, xq, yq, xs, ys, pos):
        with torch.no_grad():
            classes = ys.unique()
            one_hot = ys.view(-1, 1) == classes
            class_count = one_hot.sum(0, keepdim=True)
            yq_new = one_hot[pos].nonzero()[:, 1]

        mus = torch.mm(one_hot.t().float(), xs)
        M = mus.unsqueeze(0).repeat(len(yq), 1, 1)
        M[torch.arange(len(yq)), yq_new] -= xq

        C = class_count.repeat(len(yq), 1)
        C[torch.arange(len(yq)), yq_new] -= 1

        if self.logit_func == l2_dist:
            logit = -0.5 * (xq.unsqueeze(1) - M / C.unsqueeze(-1).clamp(min=0.1)).pow(
                2
            ).sum(-1)
        else:
            logit = (
                -(xq.unsqueeze(1) - M / C.unsqueeze(-1).clamp(min=0.1)).abs().sum(-1)
            )

        logit *= C > 0.1
        return self.xe(logit / self.T, yq_new)


class GMLoss(nn.Module):
    def __init__(self, T=1.0, logit="l2_dist", **kwargs):
        super().__init__()
        self.logit_func = logit_funcs[logit]
        self.T = T

    def forward(self, xq, yq, xs, ys, pos):
        with torch.no_grad():
            class_mask = yq.view(-1, 1) == ys.view(1, -1)
            idx = (class_mask.sum(-1) > 1).cpu()
            pos = torch.tensor(pos, device=yq.device)
            ind = torch.arange(len(pos), device=yq.device)
            class_mask[ind[idx], pos[idx]] = False

        logit = self.logit_func(xq, xs) / self.T
        logit[ind, pos] *= 0
        logit[ind[idx], pos[idx]] -= INF

        return (
            torch.logsumexp(logit, dim=-1)
            - torch.sum(logit * class_mask, dim=-1) / class_mask.sum(-1)
        ).mean()


class BCELoss(nn.BCEWithLogitsLoss):
    def __init__(self, T=1.0, logit="l2_dist", **kwargs):
        super(BCELoss, self).__init__(reduction="sum")
        self.logit_func = logit_funcs[logit]
        self.bias = nn.Parameter(torch.zeros(1))
        self.T = T

    def forward(self, xq, yq, xs, ys, pos):
        with torch.no_grad():
            one_hot_yq = yq.view(-1, 1) == ys.view(1, -1)
            pos = torch.tensor(pos, device=yq.device)
            ind = torch.arange(len(pos), device=yq.device)

        logit = self.logit_func(xq, xs) / self.T
        logit[ind, pos] *= 0
        logit[ind, pos] += INF

        return super(BCELoss, self).forward(logit, one_hot_yq.float()) / len(yq)


class AsymmetricLoss(BCELoss):
    def __init__(
        self,
        T=1.0,
        logit="l2_dist",
        gamma_neg=4,
        gamma_pos=0,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
        **kwargs,
    ):
        super(AsymmetricLoss, self).__init__(T, logit)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, xq, yq, xs, ys, pos):
        with torch.no_grad():
            one_hot_yq = yq.view(-1, 1) == ys.view(1, -1)
            pos = torch.tensor(pos, device=yq.device)
            ind = torch.arange(len(pos), device=yq.device)

        logit = self.logit_func(xq, xs) / self.T
        logit[ind, pos] *= 0
        logit[ind, pos] += INF

        x_sigmoid = torch.sigmoid(logit)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = one_hot_yq * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (~one_hot_yq) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    pt = xs_pos * one_hot_yq + xs_neg * (~one_hot_yq)
                    one_sided_gamma = self.gamma_pos * one_hot_yq + self.gamma_neg * (
                        ~one_hot_yq
                    )
                    one_sided_w = torch.pow((1 - pt).clamp(min=0), one_sided_gamma)
            else:
                pt = xs_pos * one_hot_yq + xs_neg * (~one_hot_yq)
                one_sided_gamma = self.gamma_pos * one_hot_yq + self.gamma_neg * (
                    ~one_hot_yq
                )
                one_sided_w = torch.pow((1 - pt).clamp(min=0), one_sided_gamma)
            loss *= one_sided_w

        return -loss.sum(dim=-1).mean()


class ProtoNet(nn.Module):
    def __init__(self, T=1.0, logit="l2_dist", class_aware_sampler=None, **kwargs):
        super().__init__()
        self.logit_func = logit_funcs[logit]
        assert class_aware_sampler is not None
        self.class_num = int(class_aware_sampler.split(",")[0])
        self.sample_num = int(class_aware_sampler.split(",")[1])
        self.support_num = 4
        self.query_num = self.sample_num - self.support_num
        self.xe = nn.CrossEntropyLoss()
        self.T = T

    def forward(self, xq, yq, xs, ys, pos):
        with torch.no_grad():
            sorted_indices = ys.argsort()
            y_q = torch.arange(self.class_num, device=ys.device).repeat_interleave(
                self.query_num
            )

        sorted_xs = xs[sorted_indices]
        xs_total = sorted_xs.reshape(self.class_num, self.sample_num, xs.size(-1))
        x_s = xs_total[:, : self.support_num, :]
        x_q = xs_total[:, self.support_num :, :].reshape(-1, x_s.size(-1))

        proto = x_s.mean(dim=1)
        return self.xe(self.logit_func(x_q, proto) / self.T, y_q)


class MatchingNet(nn.Module):
    def __init__(self, T=1.0, logit="l2_dist", class_aware_sampler=None, **kwargs):
        super().__init__()
        self.logit_func = logit_funcs[logit]
        assert class_aware_sampler is not None
        self.class_num = int(class_aware_sampler.split(",")[0])
        self.sample_num = int(class_aware_sampler.split(",")[1])
        self.support_num = 4
        self.query_num = self.sample_num - self.support_num
        self.xe = nn.CrossEntropyLoss()
        self.T = T

    def forward(self, xq, yq, xs, ys, pos):
        with torch.no_grad():
            sorted_indices = ys.argsort()
            y_s = torch.arange(self.class_num, device=ys.device).repeat_interleave(
                self.support_num
            )
            y_q = torch.arange(self.class_num, device=ys.device).repeat_interleave(
                self.query_num
            )
            yq_new = torch.zeros_like(y_q)
            class_mask = y_q.view(-1, 1) == y_s.view(1, -1)

        sorted_xs = xs[sorted_indices]
        xs_total = sorted_xs.reshape(self.class_num, self.sample_num, xs.size(-1))
        x_s = xs_total[:, : self.support_num, :].reshape(-1, xs.size(-1))
        x_q = xs_total[:, self.support_num :, :].reshape(-1, xs.size(-1))

        distances = self.logit_func(x_q, x_s)
        pos_logit = torch.logsumexp(
            masked_logit(distances, class_mask), 1, keepdim=True
        )
        neg_logit = torch.logsumexp(
            masked_logit(distances, ~class_mask), 1, keepdim=True
        )

        return self.xe(torch.cat([pos_logit, neg_logit], 1), yq_new)


class WrapperLoss(torch.nn.Module):
    def __init__(self, loss, class_proxy=None):
        super(WrapperLoss, self).__init__()
        self.rank = distributed.get_rank() if distributed.is_initialized() else 0
        self.loss = loss

        if class_proxy is not None:
            self.class_proxy = nn.Parameter(torch.empty(class_proxy))
            nn.init.kaiming_uniform_(self.class_proxy, a=2.23)
        else:
            self.class_proxy = None

    def forward(self, local_embeddings, local_labels):
        local_labels = local_labels.squeeze().long()
        batch_size = local_embeddings.size(0)

        embeddings = utils.gather_across_processes(local_embeddings)
        labels = utils.gather_across_processes(local_labels)

        if self.class_proxy is not None:
            embeddings = torch.cat([embeddings, self.class_proxy])
            class_labels = torch.arange(
                self.class_proxy.size(0), dtype=labels.dtype, device=labels.device
            )
            labels = torch.cat([labels, class_labels])

        pos = torch.arange(
            batch_size * self.rank, batch_size * (self.rank + 1)
        ).tolist()
        return self.loss(local_embeddings, local_labels, embeddings, labels, pos)
