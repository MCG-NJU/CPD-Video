import math

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import DistilBertModel

from .alias_multinomial import AliasMethod


class CPD(nn.Module):
    """ Cross Pair Discrimination (CPD) model with memory bank. """

    def __init__(self, visual_encoder, N_data,
                 emb_dim=256, dropout=0, K=4096, T=0.07, m=0.5, gpu=None):
        super(CPD, self).__init__()

        self.visual_encoder = visual_encoder
        self.textual_encoder = DistilBertModel.from_pretrained(
            'distilbert-base-uncased')

        self.emb_dim = emb_dim
        self.dropout = dropout
        self._prepare_base_model()

        self.vis_emb = nn.Linear(self.feature_dim, emb_dim)
        self.text_emb = nn.Sequential(nn.Linear(
            768, emb_dim*2), nn.BatchNorm1d(emb_dim*2), nn.ReLU(), nn.Linear(emb_dim*2, emb_dim))

        self.N_data = N_data
        self.K = K
        self.T = T
        self.m = m
        self.unigrams = torch.ones(N_data)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda(gpu)
        stdv = 1. / math.sqrt(emb_dim / 3)
        self.register_buffer('Z_v', torch.tensor([-1.0]))
        self.register_buffer('Z_t', torch.tensor([-1.0]))
        self.register_buffer('vis_memory', torch.rand(
            N_data, emb_dim).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('text_memory', torch.rand(
            N_data, emb_dim).mul_(2 * stdv).add_(-stdv))

    def _prepare_base_model(self):
        self.feature_dim = getattr(self.visual_encoder, 'fc').in_features
        if self.dropout == 0:
            setattr(self.visual_encoder, 'fc', Identity())
        else:
            setattr(self.visual_encoder, 'fc', nn.Dropout(p=self.dropout))

    def forward(self, v, t, idx=None, updbert=False):

        # Extract visual embedding
        bs = v.size(0)
        vis_base_out = self.visual_encoder(v.squeeze())
        vis_base_out = self.vis_emb(vis_base_out)
        vis_base_out = nn.functional.normalize(vis_base_out, dim=1)

        # Extract textual embedding
        mask = (t != 0).float().unsqueeze(2)
        text_base_out = self.forward_text_encoder(t, updbert)
        # average pooling
        text_base_out = (text_base_out * mask).sum(1) / mask.sum(1)
        text_base_out = self.text_emb(text_base_out)
        text_base_out = nn.functional.normalize(text_base_out, dim=1)

        if idx is None:
            # validation forward
            return vis_base_out, text_base_out

        # Video and text pair discrimination
        slct_idx = self.multinomial.draw(bs * (self.K + 1)).view(bs, -1)
        slct_idx.select(1, 0).copy_(idx.data)

        text_feats = torch.index_select(
            self.text_memory, 0, slct_idx.view(-1)).detach()
        text_feats = text_feats.view(bs, self.K+1, self.emb_dim)
        vis_out = torch.bmm(text_feats, vis_base_out.view(bs, self.emb_dim, 1))
        vis_out = torch.exp(torch.div(vis_out, self.T))

        vis_feats = torch.index_select(
            self.vis_memory, 0, slct_idx.view(-1)).detach()
        vis_feats = vis_feats.view(bs, self.K+1, self.emb_dim)
        text_out = torch.bmm(
            vis_feats, text_base_out.view(bs, self.emb_dim, 1))
        text_out = torch.exp(torch.div(text_out, self.T))

        with torch.no_grad():
            if self.Z_v[0].item() < 0:
                self.Z_v[0] = vis_out.mean() * self.N_data
                self.Z_v[0] = reduce_tensor(self.Z_v[0])
                print('normalization constant Z_v is set to {:.1f}'.format(
                    self.Z_v[0].item()))
            if self.Z_t[0].item() < 0:
                self.Z_t[0] = text_out.mean() * self.N_data
                self.Z_t[0] = reduce_tensor(self.Z_t[0])
                print('normalization constant Z_t is set to {:.1f}'.format(
                    self.Z_t[0].item()))
        vis_out = torch.div(vis_out, self.Z_v.detach())
        text_out = torch.div(text_out, self.Z_t.detach())

        self.update_memory(vis_base_out, text_base_out, idx)
        return vis_out, text_out

    def update_memory(self, vis_feat, text_feat, idx):
        vis_feat = concat_all_gather(vis_feat)
        text_feat = concat_all_gather(text_feat)
        idx = concat_all_gather(idx)
        with torch.no_grad():
            vis_pos = torch.index_select(self.vis_memory, 0, idx.view(-1))
            vis_pos.mul_(self.m).add_(torch.mul(vis_feat, 1 - self.m))
            vis_update = nn.functional.normalize(vis_pos, dim=1)
            self.vis_memory.index_copy_(0, idx, vis_update)

            text_pos = torch.index_select(self.text_memory, 0, idx.view(-1))
            text_pos.mul_(self.m).add_(torch.mul(text_feat, 1 - self.m))
            text_update = nn.functional.normalize(text_pos, dim=1)
            self.text_memory.index_copy_(0, idx, text_update)

    def forward_text_encoder(self, t, updbert):
        """ 
            Curriculum learning for CPD training stage.
            Freezing textual encoder in stage I. 
        """
        if updbert:
            text_out = self.textual_encoder(t)[0]
        else:
            with torch.no_grad():
                text_out = self.textual_encoder(t)[0]
        return text_out


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


def get_fine_tuning_parameters(model, lr):
    """Set small learning rate on bert"""

    parameters = []
    for k, v in model.named_parameters():
        if 'textual_encoder' in k:
            parameters.append({'params': v, 'lr': 0})
        else:
            parameters.append({'params': v, 'lr': lr})
    return parameters


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: dist.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def reduce_tensor(tensor):
    output = tensor.clone()
    dist.all_reduce(output, op=dist.ReduceOp.SUM)
    output /= dist.get_world_size()
    return output
