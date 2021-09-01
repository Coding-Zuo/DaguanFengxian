# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss(https://arxiv.org/pdf/1708.02002.pdf)
    Shape:
        - input: (N, C)
        - target: (N)
        - Output: Scalar loss
    Examples:
        # >>> loss = FocalLoss(gamma=2, alpha=[1.0]*7)
        # >>> input = torch.randn(3, 7, requires_grad=True)
        # >>> target = torch.empty(3, dtype=torch.long).random_(7)
        # >>> output = loss(input, target)
        # >>> output.backward()
    """

    def __init__(self, gamma=0, alpha=None, reduction="none"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.FloatTensor(alpha)
            else:
                self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        """
        - input: (N, C), logits
        - target: (N)
        - Output: Scalar loss
        """
        # [N, 1]
        target = target.unsqueeze(-1)
        # [N,C]
        pt = F.softmax(input, dim=-1)
        logpt = F.log_softmax(input, dim=-1)

        # 得到答案标签所获得 概率值 和 对数概率值
        # [N]
        pt = pt.gather(1, target).squeeze(-1)
        logpt = logpt.gather(1, target).squeeze(-1)

        # 加上class weight
        if self.alpha is not None:
            # [N] at[i]=alpha[target[i]]
            # 得到每个样本应该得到的class weight
            at = self.alpha.gather(0, target.squeeze(-1))
            logpt = logpt * at.to(logpt.device)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction == 'none':
            return loss
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

    def __str__(self):
        return f"Focal Loss gamma:{self.gamma}"

    def __repr__(self):
        return str(self)

    @staticmethod
    def convert_binary_pred_to_two_dimension(x, is_logits=True):
        """
        Args:
            x: (*): (log) prob of some instance has label 1
            is_logits: if True, x represents log prob; otherwhise presents prob
        Returns:
            y: (*, 2), where y[*, 1] == log prob of some instance has label 0,
                             y[*, 0] = log prob of some instance has label 1
        """
        probs = torch.sigmoid(x) if is_logits else x
        probs = probs.unsqueeze(-1)
        probs = torch.cat([1 - probs, probs], dim=-1)
        logprob = torch.log(probs + 1e-4)  # # 1e-4 to prevent being rounded to 0 in fp16
        return logprob


class FocalLoss1(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, device=None):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1).to(device)
        else:
            self.alpha = alpha.to(device)

        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = torch.zeros_like(inputs).to(inputs.device)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print("class_mask: ", class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        # print("alpha: ", alpha)

        probs = (P * class_mask).sum(1).view(-1, 1)
        # print("probs: ", probs)

        log_p = probs.log()
        # print('log_p size= {}'.format(log_p.size()))
        # print(log_p)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
