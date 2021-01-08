import torch
from torch import nn

eps = 1e-7

class NCECriterion(nn.Module):

    def __init__(self, n_data):
        super(NCECriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        batchsize = x.shape[0]
        m = x.shape[1] - 1

        Pn = 1 / float(self.n_data)

        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss_pos = - log_D1.sum(0) / batchsize
        loss_neg = - log_D0.view(-1, 1).sum(0) / batchsize
        return loss_pos, loss_neg
