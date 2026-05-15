# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any, Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import FocalLoss, VarifocalLoss
from ultralytics.utils.metrics import bbox_iou

from .ops import HungarianMatcher

#################################################################################################
def get_inner_iou(box1, box2, xywh=True, eps=1e-7, ratio=0.7):
    if not xywh:
        box1, box2 = ops.xyxy2xywh(box1), ops.xyxy2xywh(box2)
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - (w1 * ratio) / 2, x1 + (w1 * ratio) / 2, y1 - (h1 * ratio) / 2, y1 + (
                h1 * ratio) / 2
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - (w2 * ratio) / 2, x2 + (w2 * ratio) / 2, y2 - (h2 * ratio) / 2, y2 + (
                h2 * ratio) / 2

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 * ratio * ratio + w2 * h2 * ratio * ratio - inter + eps
    return inter / union


class WiseIouLoss(torch.nn.Module):
    ''' :param monotonous: {
            None: origin V1
            True: monotonic FM V2
            False: non-monotonic FM V3
        }'''
    momentum = 1e-2
    alpha = 1.7
    delta = 2.7

    def __init__(self, ltype='WIoU', monotonous=False, inner_iou=False, focaler_iou=False):
        super().__init__()
        assert getattr(self, f'_{ltype}', None), f'The loss function {ltype} does not exist'
        self.ltype = ltype
        self.monotonous = monotonous
        self.inner_iou = inner_iou
        self.focaler_iou = focaler_iou
        self.register_buffer('iou_mean', torch.tensor(1.))

    def __getitem__(self, item):
        if callable(self._fget[item]):
            self._fget[item] = self._fget[item]()
        return self._fget[item]

    def forward(self, pred, target, ret_iou=False, ratio=1.0, d=0.0, u=0.95, **kwargs):
        self._fget = {
            # pred, target: x0,y0,x1,y1
            'pred': self._xywh2xyxy(pred),
            'target': self._xywh2xyxy(target),
            # x,y,w,h
            'pred_xy': pred[..., :2],
            'pred_wh': pred[..., 2:],
            'target_xy': target[..., :2],
            'target_wh': target[..., 2:],
            # x0,y0,x1,y1
            'min_coord': lambda: torch.minimum(self['pred'][..., :4], self['target'][..., :4]),
            'max_coord': lambda: torch.maximum(self['pred'][..., :4], self['target'][..., :4]),
            # The overlapping region
            'wh_inter': lambda: torch.relu(self['min_coord'][..., 2: 4] - self['max_coord'][..., :2]),
            's_inter': lambda: torch.prod(self['wh_inter'], dim=-1),
            # The area covered
            's_union': lambda: torch.prod(self['pred_wh'], dim=-1) +
                               torch.prod(self['target_wh'], dim=-1) - self['s_inter'],
            # The smallest enclosing box
            'wh_box': lambda: self['max_coord'][..., 2: 4] - self['min_coord'][..., :2],
            's_box': lambda: torch.prod(self['wh_box'], dim=-1),
            'l2_box': lambda: torch.square(self['wh_box']).sum(dim=-1),
            # The central points' connection of the bounding boxes
            'd_center': lambda: self['pred_xy'] - self['target_xy'],
            'l2_center': lambda: torch.square(self['d_center']).sum(dim=-1),
            # IoU / Inner-IoU / Focaler-IoU
            'iou': lambda: (1 - get_inner_iou(pred, target, xywh=False, ratio=ratio).squeeze()) if self.inner_iou else (
                1 - ((self['s_inter'] / self['s_union'] - d) / (u - d)).clamp(0, 1) if self.focaler_iou else 1 - self[
                    's_inter'] / self['s_union']),
        }

        if self.training:
            self.iou_mean = self.iou_mean.to(self['iou'].device)
            self.iou_mean.mul_(1 - self.momentum)
            self.iou_mean.add_(self.momentum * self['iou'].detach().mean())

        ret = self._scaled_loss(getattr(self, f'_{self.ltype}')(**kwargs)), self['iou']
        delattr(self, '_fget')
        return ret if ret_iou else ret[0]

    def _scaled_loss(self, loss, iou=None):
        if isinstance(self.monotonous, bool):
            beta = (self['iou'].detach() if iou is None else iou) / self.iou_mean

            if self.monotonous:
                loss *= beta.sqrt()
            else:
                divisor = self.delta * torch.pow(self.alpha, beta - self.delta)
                loss *= beta / divisor
        return loss

    def _IoU(self):
        return self['iou']

    def _WIoU(self):
        dist = torch.exp(self['l2_center'] / self['l2_box'].detach())
        return dist * self['iou']

    def _EIoU(self):
        penalty = self['l2_center'] / self['l2_box'] \
                  + torch.square(self['d_center'] / self['wh_box']).sum(dim=-1)
        return self['iou'] + penalty

    def _GIoU(self):
        return self['iou'] + (self['s_box'] - self['s_union']) / self['s_box']

    def _DIoU(self):
        return self['iou'] + self['l2_center'] / self['l2_box']

    def _CIoU(self, eps=1e-4):
        v = 4 / math.pi ** 2 * \
            (torch.atan(self['pred_wh'][..., 0] / (self['pred_wh'][..., 1] + eps)) -
             torch.atan(self['target_wh'][..., 0] / (self['target_wh'][..., 1] + eps))) ** 2
        alpha = v / (self['iou'] + v)
        return self['iou'] + self['l2_center'] / self['l2_box'] + alpha.detach() * v

    def _SIoU(self, theta=4):
        # Angle Cost
        angle = torch.arcsin(torch.abs(self['d_center']).min(dim=-1)[0] / (self['l2_center'].sqrt() + 1e-4))
        angle = torch.sin(2 * angle) - 2
        # Dist Cost
        dist = angle[..., None] * torch.square(self['d_center'] / self['wh_box'])
        dist = 2 - torch.exp(dist[..., 0]) - torch.exp(dist[..., 1])
        # Shape Cost
        d_shape = torch.abs(self['pred_wh'] - self['target_wh'])
        big_shape = torch.maximum(self['pred_wh'], self['target_wh'])
        w_shape = 1 - torch.exp(- d_shape[..., 0] / big_shape[..., 0])
        h_shape = 1 - torch.exp(- d_shape[..., 1] / big_shape[..., 1])
        shape = w_shape ** theta + h_shape ** theta
        return self['iou'] + (dist + shape) / 2

    def _MPDIoU(self, mpdiou_hw):
        d1 = (self['target'][..., 0] - self['pred'][..., 0]) ** 2 + (self['target'][..., 1] - self['pred'][..., 1]) ** 2
        d2 = (self['target'][..., 2] - self['pred'][..., 2]) ** 2 + (self['target'][..., 3] - self['pred'][..., 3]) ** 2
        return self['iou'] + d1 / mpdiou_hw + d2 / mpdiou_hw

    def _ShapeIoU(self, scale=0.0):
        b1_x1, b1_y1, b1_x2, b1_y2 = self['pred'].chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = self['target'].chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-7
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-7

        # Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance
        ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
        hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        c2 = cw ** 2 + ch ** 2 + 1e-7  # convex diagonal squared
        center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
        center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        center_distance = hh * center_distance_x + ww * center_distance_y
        distance = center_distance / c2

        # Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape
        omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        return self['iou'] + distance.squeeze() + 0.5 * shape_cost.squeeze()

    def _PIoU(self):
        b1_x1, b1_y1, b1_x2, b1_y2 = self['pred'].chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = self['target'].chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-7
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-7

        dw1 = torch.abs(b1_x2.minimum(b1_x1) - b2_x2.minimum(b2_x1))
        dw2 = torch.abs(b1_x2.maximum(b1_x1) - b2_x2.maximum(b2_x1))
        dh1 = torch.abs(b1_y2.minimum(b1_y1) - b2_y2.minimum(b2_y1))
        dh2 = torch.abs(b1_y2.maximum(b1_y1) - b2_y2.maximum(b2_y1))
        P = ((dw1 + dw2) / torch.abs(w2) + (dh1 + dh2) / torch.abs(h2)) / 4
        piou_v1 = self['iou'] - torch.exp(-P.squeeze() ** 2) + 1
        return piou_v1

    def _PIoU2(self, Lambda=1.3):
        b1_x1, b1_y1, b1_x2, b1_y2 = self['pred'].chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = self['target'].chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-7
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-7

        dw1 = torch.abs(b1_x2.minimum(b1_x1) - b2_x2.minimum(b2_x1))
        dw2 = torch.abs(b1_x2.maximum(b1_x1) - b2_x2.maximum(b2_x1))
        dh1 = torch.abs(b1_y2.minimum(b1_y1) - b2_y2.minimum(b2_y1))
        dh2 = torch.abs(b1_y2.maximum(b1_y1) - b2_y2.maximum(b2_y1))
        P = ((dw1 + dw2) / torch.abs(w2) + (dh1 + dh2) / torch.abs(h2)) / 4
        piou_v1 = self['iou'] - torch.exp(-P.squeeze() ** 2) + 1
        q = torch.exp(-P.squeeze())
        x = q * Lambda
        return 3 * x * torch.exp(-x ** 2) * piou_v1

    def _xywh2xyxy(self, data):
        x1, y1, w1, h1 = data.chunk(4, -1)
        w1_, h1_ = w1 / 2, h1 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        data = torch.cat([b1_x1, b1_y1, b1_x2, b1_y2], dim=-1)
        return data

    def __repr__(self):
        return f'{self.__name__}(iou_mean={self.iou_mean.item():.3f})'

    __name__ = property(lambda self: self.ltype)


def wasserstein_loss(box1, box2, xywh=True, eps=1e-7, constant=12.8):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.
    Code is modified from https://github.com/Zzh-tju/CIoU.
    Args:
        pred (Tensor): Predicted bboxes of format (x_min, y_min, x_max, y_max),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1, h1, w2, h2 = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1, x1 + w1, y1 - h1, y1 + h1
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2, x2 + w2, y2 - h2, y2 + h2
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    b1_x_center, b1_y_center = b1_x1 + w1 / 2, b1_y1 + h1 / 2
    b2_x_center, b2_y_center = b2_x1 + w2 / 2, b2_y1 + h2 / 2
    center_distance = (b1_x_center - b2_x_center) ** 2 + (b1_y_center - b2_y_center) ** 2 + eps
    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wasserstein_2 = center_distance + wh_distance
    return torch.exp(-torch.sqrt(wasserstein_2) / constant)

def bbox_focaler_mpdiou(box1, box2, xywh=True, mpdiou_hw=1, eps=1e-7, d=0.0, u=0.95):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
                (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    # Focaler-IoU
    iou = ((iou - d) / (u - d)).clamp(0, 1)  # default d=0.00, u=0.95
    d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
    d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
    return iou - d1 / mpdiou_hw - d2 / mpdiou_hw  # MPDIoU

def bbox_inner_mpdiou(box1, box2, xywh=True, mpdiou_hw=2, ratio=0.7, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Inner-IoU
    innner_iou = get_inner_iou(box1, box2, xywh=xywh, ratio=ratio)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
                (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
    d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
    return innner_iou - d1 / mpdiou_hw - d2 / mpdiou_hw  # MPDIoU

def bbox_mpdiou(box1, box2, xywh=True, mpdiou_hw=2, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
                (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
    d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
    return iou - d1 / mpdiou_hw - d2 / mpdiou_hw  # MPDIoU

def bbox_focaler_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, ShapeIoU=False, PIoU=False, PIoU2=False, eps=1e-7, scale=0.0, d=0.0, u=0.95, Lambda=1.3):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        EIoU (bool, optional): If True, calculate Efficient IoU. Defaults to False.
        SIoU (bool, optional): If True, calculate Scylla IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    # Focaler-IoU
    iou = ((iou - d) / (u - d)).clamp(0, 1)  # default d=0.00, u=0.95
    if CIoU or DIoU or GIoU or EIoU or SIoU or ShapeIoU or PIoU or PIoU2:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or PIoU or PIoU2 or ShapeIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2) # EIoU
            elif SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return iou - 0.5 * (distance_cost + shape_cost) + eps # SIoU
            elif ShapeIoU:
                #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance
                ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
                ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
                c2 = cw ** 2 + ch ** 2 + eps                            # convex diagonal squared
                center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
                center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
                center_distance = hh * center_distance_x + ww * center_distance_y
                distance = center_distance / c2

                #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape
                omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return iou - distance - 0.5 * shape_cost
            elif PIoU or PIoU2:
                dw1 = torch.abs(b1_x2.minimum(b1_x1)-b2_x2.minimum(b2_x1))
                dw2 = torch.abs(b1_x2.maximum(b1_x1)-b2_x2.maximum(b2_x1))
                dh1 = torch.abs(b1_y2.minimum(b1_y1)-b2_y2.minimum(b2_y1))
                dh2 = torch.abs(b1_y2.maximum(b1_y1)-b2_y2.maximum(b2_y1))
                P = ((dw1+dw2)/torch.abs(w2)+(dh1+dh2)/torch.abs(h2))/4
                piou_v1 = 1 - iou - torch.exp(-P**2) + 1
                if PIoU:
                    return 1 - piou_v1
                elif PIoU2:
                    q=torch.exp(-P)
                    x=q*Lambda
                    return 1 - 3*x*torch.exp(-x**2)*piou_v1
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU
    ##############################################################################################
class DETRLoss(nn.Module):
    """
    DETR (DEtection TRansformer) Loss class for calculating various loss components.

    This class computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary losses for the
    DETR object detection model.

    Attributes:
        nc (int): Number of classes.
        loss_gain (Dict[str, float]): Coefficients for different loss components.
        aux_loss (bool): Whether to compute auxiliary losses.
        use_fl (bool): Whether to use FocalLoss.
        use_vfl (bool): Whether to use VarifocalLoss.
        use_uni_match (bool): Whether to use a fixed layer for auxiliary branch label assignment.
        uni_match_ind (int): Index of fixed layer to use if use_uni_match is True.
        matcher (HungarianMatcher): Object to compute matching cost and indices.
        fl (FocalLoss | None): Focal Loss object if use_fl is True, otherwise None.
        vfl (VarifocalLoss | None): Varifocal Loss object if use_vfl is True, otherwise None.
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(
        self,
        nc: int = 80,
        loss_gain: Optional[Dict[str, float]] = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        use_vfl: bool = False,
        use_uni_match: bool = False,
        uni_match_ind: int = 0,
        gamma: float = 1.5,
        alpha: float = 0.25,
        loss_cls: Optional[str] = None,
        loss_bbox: str = "l1",
        loss_giou: str = "giou",
    ):
        """
        Initialize DETR loss function with customizable components and gains.

        Uses default loss_gain if not provided. Initializes HungarianMatcher with preset cost gains. Supports auxiliary
        losses and various loss types.

        Args:
            nc (int): Number of classes.
            loss_gain (Dict[str, float], optional): Coefficients for different loss components.
            aux_loss (bool): Whether to use auxiliary losses from each decoder layer.
            use_fl (bool): Whether to use FocalLoss.
            use_vfl (bool): Whether to use VarifocalLoss.
            use_uni_match (bool): Whether to use fixed layer for auxiliary branch label assignment.
            uni_match_ind (int): Index of fixed layer for uni_match.
            gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
            alpha (float): The balancing factor used to address class imbalance.
            loss_cls (str | None): Explicit classification loss type. Uses legacy use_fl/use_vfl defaults when None.
            loss_bbox (str): Bounding box L1 loss switch, choices are 'l1' and 'none'.
            loss_giou (str): GIoU loss switch, choices are 'giou' and 'none'.
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1}
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.loss_cls_type = self._resolve_loss_cls(loss_cls, use_fl, use_vfl)
        self.loss_bbox_type = self._validate_loss_option("loss_bbox", loss_bbox, {"l1", "none"})
        self.loss_giou_type = self._validate_loss_option("loss_giou", loss_giou, {"giou", "none"})
        self.loss_bbox_enabled = self.loss_bbox_type != "none"
        self.loss_giou_enabled = self.loss_giou_type != "none"
        self._legacy_varifocal_fallback = loss_cls is None and use_vfl and use_fl
        self.fl = FocalLoss(gamma, alpha) if self.loss_cls_type == "focal" or self._legacy_varifocal_fallback else None
        self.vfl = VarifocalLoss(gamma, alpha) if self.loss_cls_type == "varifocal" else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

    @staticmethod
    def _validate_loss_option(name: str, value: str, valid: set[str]) -> str:
        """Validate a DETR loss switch option."""
        if not isinstance(value, str):
            raise ValueError(f"Unknown {name}: {value}. Valid: {sorted(valid)}")
        normalized = value.lower()
        if normalized not in valid:
            raise ValueError(f"Unknown {name}: {normalized}. Valid: {sorted(valid)}")
        return normalized

    @classmethod
    def _resolve_loss_cls(cls, loss_cls: Optional[str], use_fl: bool, use_vfl: bool) -> str:
        """Resolve DETR classification loss while preserving legacy defaults when unspecified."""
        if loss_cls is None:
            if use_vfl:
                return "varifocal"
            if use_fl:
                return "focal"
            return "bce"
        return cls._validate_loss_option("loss_cls", loss_cls, {"bce", "focal", "varifocal"})

    def _get_loss_class(
        self, pred_scores: torch.Tensor, targets: torch.Tensor, gt_scores: torch.Tensor, num_gts: int, postfix: str = ""
    ) -> Dict[str, torch.Tensor]:
        """
        Compute classification loss based on predictions, target values, and ground truth scores.

        Args:
            pred_scores (torch.Tensor): Predicted class scores with shape (B, N, C).
            targets (torch.Tensor): Target class indices with shape (B, N).
            gt_scores (torch.Tensor): Ground truth confidence scores with shape (B, N).
            num_gts (int): Number of ground truth objects.
            postfix (str, optional): String to append to the loss name for identification in multi-loss scenarios.

        Returns:
            (Dict[str, torch.Tensor]): Dictionary containing classification loss value.

        Notes:
            The function supports different classification loss types:
            - Varifocal Loss when configured, with legacy Focal fallback only for empty-GT batches.
            - Focal Loss when configured.
            - BCE Loss when configured.
        """
        # Logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]
        # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, num_classes)
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if self.loss_cls_type == "varifocal":
            if num_gts or not self._legacy_varifocal_fallback:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        elif self.loss_cls_type == "focal":
            loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, postfix: str = ""
    ) -> Dict[str, torch.Tensor]:
        """
        Compute bounding box and GIoU losses for predicted and ground truth bounding boxes.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape (N, 4).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (N, 4).
            postfix (str, optional): String to append to the loss names for identification in multi-loss scenarios.

        Returns:
            (Dict[str, torch.Tensor]): Dictionary containing:
                - loss_bbox{postfix}: L1 loss between predicted and ground truth boxes, scaled by the bbox loss gain.
                - loss_giou{postfix}: GIoU loss between predicted and ground truth boxes, scaled by the giou loss gain.

        Notes:
            If no ground truth boxes are provided (empty list), zero-valued tensors are returned for both losses.
        """
        # Boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        if self.loss_bbox_enabled:
            loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
        else:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)

        if self.loss_giou_enabled:
            #loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
            # CIOU
            #loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, CIoU=True)
            ##############################################################################################################
            # 🌟 选项B：动态加权融合 (0.5 GIoU + 0.5 NWD)
            # 1. 算出原生 GIoU (保留大目标抗视差能力)
            #raw_giou = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
            # 2. 算出 NWD (专攻微小目标)
            #raw_nwd = 1.0 - wasserstein_loss(pred_bboxes, gt_bboxes, xywh=True)
            # 3. 按比例加权赋值给 loss[name_giou]
            #loss[name_giou] = 0.5 * raw_giou + 0.5 * raw_nwd
            ##############################################################################################################
            # 使用 DIoU：只约束中心点距离，不强制约束长宽比，完美避开多模态视差冲突
            #loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, DIoU=True)
            ##############################################################################################################
            # 方案二：Focaler-GIoU (动态梯度聚焦，不改变底层几何包容性)
            loss[name_giou] = 1.0 - bbox_focaler_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True, d=0.0, u=0.95)
            ##############################################################################################################
            # 方案三：MPDIoU (左上角+右下角两点约束，对齐更加平滑)
            #loss[name_giou] = 1.0 - bbox_mpdiou(pred_bboxes, gt_bboxes, xywh=True, mpdiou_hw=2)
            ##############################################################################################################
            # loss[name_giou] = 1.0 - bbox_inner_iou(pred_bboxes, gt_bboxes, xywh=True, SIoU=True,ratio=1.25)  # Inner IoU
            # loss[name_giou] = 1.0 - bbox_inner_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True, ratio=0.7) # Inner IoU
            # loss[name_giou] = 1.0 - bbox_focaler_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True, d=0.0, u=0.95) # Focaler IoU
            # loss[name_giou] = 1.0 - bbox_mpdiou(pred_bboxes, gt_bboxes, xywh=True, mpdiou_hw=2) # MPDIoU
            # loss[name_giou] = 1.0 - bbox_inner_mpdiou(pred_bboxes, gt_bboxes, xywh=True, mpdiou_hw=2, ratio=0.7) # Inner-MPDIoU
            # loss[name_giou] = 1.0 - bbox_focaler_mpdiou(pred_bboxes, gt_bboxes, xywh=True, mpdiou_hw=2, d=0.0, u=0.95) # Focaler-MPDIoU
            loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
            loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
        else:
            loss[name_giou] = torch.tensor(0.0, device=self.device)
        return {k: v.squeeze() for k, v in loss.items()}

    # This function is for future RT-DETR Segment models
    # def _get_loss_mask(self, masks, gt_mask, match_indices, postfix=''):
    #     # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
    #     name_mask = f'loss_mask{postfix}'
    #     name_dice = f'loss_dice{postfix}'
    #
    #     loss = {}
    #     if sum(len(a) for a in gt_mask) == 0:
    #         loss[name_mask] = torch.tensor(0., device=self.device)
    #         loss[name_dice] = torch.tensor(0., device=self.device)
    #         return loss
    #
    #     num_gts = len(gt_mask)
    #     src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices)
    #     src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
    #     # TODO: torch does not have `sigmoid_focal_loss`, but it's not urgent since we don't use mask branch for now.
    #     loss[name_mask] = self.loss_gain['mask'] * F.sigmoid_focal_loss(src_masks, target_masks,
    #                                                                     torch.tensor([num_gts], dtype=torch.float32))
    #     loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
    #     return loss

    # This function is for future RT-DETR Segment models
    # @staticmethod
    # def _dice_loss(inputs, targets, num_gts):
    #     inputs = F.sigmoid(inputs).flatten(1)
    #     targets = targets.flatten(1)
    #     numerator = 2 * (inputs * targets).sum(1)
    #     denominator = inputs.sum(-1) + targets.sum(-1)
    #     loss = 1 - (numerator + 1) / (denominator + 1)
    #     return loss.sum() / num_gts

    def _get_loss_aux(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: List[int],
        match_indices: Optional[List[Tuple]] = None,
        postfix: str = "",
        masks: Optional[torch.Tensor] = None,
        gt_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get auxiliary losses for intermediate decoder layers.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes from auxiliary layers.
            pred_scores (torch.Tensor): Predicted scores from auxiliary layers.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            gt_cls (torch.Tensor): Ground truth classes.
            gt_groups (List[int]): Number of ground truths per image.
            match_indices (List[Tuple], optional): Pre-computed matching indices.
            postfix (str, optional): String to append to loss names.
            masks (torch.Tensor, optional): Predicted masks if using segmentation.
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.

        Returns:
            (Dict[str, torch.Tensor]): Dictionary of auxiliary losses.
        """
        # NOTE: loss class, bbox, giou, mask, dice
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask,
            )
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=aux_masks,
                gt_mask=gt_mask,
                postfix=postfix,
                match_indices=match_indices,
            )
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]
            # if masks is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
            #     loss[3] += loss_[f'loss_mask{postfix}']
            #     loss[4] += loss_[f'loss_dice{postfix}']

        loss = {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
        }
        # if masks is not None and gt_mask is not None:
        #     loss[f'loss_mask_aux{postfix}'] = loss[3]
        #     loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss

    @staticmethod
    def _get_index(match_indices: List[Tuple]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Extract batch indices, source indices, and destination indices from match indices.

        Args:
            match_indices (List[Tuple]): List of tuples containing matched indices.

        Returns:
            batch_idx (Tuple[torch.Tensor, torch.Tensor]): Tuple containing (batch_idx, src_idx).
            dst_idx (torch.Tensor): Destination indices.
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, match_indices: List[Tuple]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign predicted bounding boxes to ground truth bounding boxes based on match indices.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            match_indices (List[Tuple]): List of tuples containing matched indices.

        Returns:
            pred_assigned (torch.Tensor): Assigned predicted bounding boxes.
            gt_assigned (torch.Tensor): Assigned ground truth bounding boxes.
        """
        pred_assigned = torch.cat(
            [
                t[i] if len(i) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (i, _) in zip(pred_bboxes, match_indices)
            ]
        )
        gt_assigned = torch.cat(
            [
                t[j] if len(j) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (_, j) in zip(gt_bboxes, match_indices)
            ]
        )
        return pred_assigned, gt_assigned

    def _get_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: List[int],
        masks: Optional[torch.Tensor] = None,
        gt_mask: Optional[torch.Tensor] = None,
        postfix: str = "",
        match_indices: Optional[List[Tuple]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate losses for a single prediction layer.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes.
            pred_scores (torch.Tensor): Predicted class scores.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            gt_cls (torch.Tensor): Ground truth classes.
            gt_groups (List[int]): Number of ground truths per image.
            masks (torch.Tensor, optional): Predicted masks if using segmentation.
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.
            postfix (str, optional): String to append to loss names.
            match_indices (List[Tuple], optional): Pre-computed matching indices.

        Returns:
            (Dict[str, torch.Tensor]): Dictionary of losses.
        """
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
            )

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        return {
            **self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix),
            **self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix),
            # **(self._get_loss_mask(masks, gt_mask, match_indices, postfix) if masks is not None and gt_mask is not None else {})
        }

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: Dict[str, Any],
        postfix: str = "",
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate loss for predicted bounding boxes and scores.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape (L, B, N, 4).
            pred_scores (torch.Tensor): Predicted class scores, shape (L, B, N, C).
            batch (Dict[str, Any]): Batch information containing cls, bboxes, and gt_groups.
            postfix (str, optional): Postfix for loss names.
            **kwargs (Any): Additional arguments, may include 'match_indices'.

        Returns:
            (Dict[str, torch.Tensor]): Computed losses, including main and auxiliary (if enabled).

        Notes:
            Uses last elements of pred_bboxes and pred_scores for main loss, and the rest for auxiliary losses if
            self.aux_loss is True.
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get("match_indices", None)
        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]

        total_loss = self._get_loss(
            pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups, postfix=postfix, match_indices=match_indices
        )

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices, postfix
                )
            )

        return total_loss


class RTDETRDetectionLoss(DETRLoss):
    """
    Real-Time DeepTracker (RT-DETR) Detection Loss class that extends the DETRLoss.

    This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as
    an additional denoising training loss when provided with denoising metadata.
    """

    def __init__(self, model=None, **kwargs):
        """Initialize RT-DETR loss and optionally read switchable loss args from the model."""
        if model is not None:
            model_args = getattr(model, "args", None)
            kwargs.setdefault("nc", getattr(model, "nc", model.model[-1].nc))
            kwargs.setdefault("use_vfl", True)
            kwargs.setdefault("loss_cls", self._get_model_arg(model_args, "loss_cls", None))
            kwargs.setdefault("loss_bbox", self._get_model_arg(model_args, "loss_bbox", "l1"))
            kwargs.setdefault("loss_giou", self._get_model_arg(model_args, "loss_giou", "giou"))
            kwargs.setdefault("aux_loss", self._get_model_arg(model_args, "loss_aux", True))
        super().__init__(**kwargs)

    @staticmethod
    def _get_model_arg(model_args: Any, name: str, default: Any) -> Any:
        """Read a configuration value from either a namespace-style or dict-style args object."""
        if isinstance(model_args, dict):
            return model_args.get(name, default)
        return getattr(model_args, name, default)

    def forward(
        self,
        preds: Tuple[torch.Tensor, torch.Tensor],
        batch: Dict[str, Any],
        dn_bboxes: Optional[torch.Tensor] = None,
        dn_scores: Optional[torch.Tensor] = None,
        dn_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute detection loss with optional denoising loss.

        Args:
            preds (Tuple[torch.Tensor, torch.Tensor]): Tuple containing predicted bounding boxes and scores.
            batch (Dict[str, Any]): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes.
            dn_scores (torch.Tensor, optional): Denoising scores.
            dn_meta (Dict[str, Any], optional): Metadata for denoising.

        Returns:
            (Dict[str, torch.Tensor]): Dictionary containing total loss and denoising loss if applicable.
        """
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

            # Compute the denoising training loss
            dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(
        dn_pos_idx: List[torch.Tensor], dn_num_group: int, gt_groups: List[int]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get match indices for denoising.

        Args:
            dn_pos_idx (List[torch.Tensor]): List of tensors containing positive indices for denoising.
            dn_num_group (int): Number of denoising groups.
            gt_groups (List[int]): List of integers representing number of ground truths per image.

        Returns:
            (List[Tuple[torch.Tensor, torch.Tensor]]): List of tuples containing matched indices for denoising.
        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), (
                    f"Expected the same length, but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
                )
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices
