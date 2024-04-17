from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import math


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def weighted_Features(images, Q):
    """
    Q: (batch, queryL, d)
    K: (batch, sourceL, d)
    return (batch, queryL, sourceL)
    """
    batch_size, queryL = Q.size(0), Q.size(1)
    batch_size, sourceL = images.size(0), images.size(1)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    queryT = torch.transpose(Q, 1, 2)
    # sourceT = torch.transpose(images, 1, 2)

    attn = torch.bmm(images, queryT)
    # attn1 = torch.bmm(Q, sourceT)
    attn1 = torch.transpose(attn, 1, 2)

    attn = nn.LeakyReLU(0.1)(attn)
    attn1 = nn.LeakyReLU(0.1)(attn1)

    attn = l2norm(attn, 2)
    attn1 = l2norm(attn1, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn1 = torch.transpose(attn1, 1, 2).contiguous()

    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn1 = attn1.view(batch_size * sourceL, queryL)
    # Notice  Flickr30K: 9.0, 20.0;  MS-COCO: 10.0, 10.0
    attn = nn.Softmax(dim=1)(attn * 10.0)
    attn1 = nn.Softmax(dim=1)(attn1 * 10.0)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    attn1 = attn1.view(batch_size, sourceL, queryL)

    attnT = torch.transpose(attn, 1, 2)
    attn1T = torch.transpose(attn1, 1, 2)

    imagesT = torch.transpose(images, 1, 2)
    QT = torch.transpose(Q, 1, 2)

    weighted_textual_features = torch.bmm(imagesT, attnT)
    weighted_visual_features = torch.bmm(QT, attn1T)

    weighted_textual_featuresT = torch.transpose(weighted_textual_features, 1, 2)
    weighted_visual_featuresT = torch.transpose(weighted_visual_features, 1, 2)

    return weighted_textual_featuresT, weighted_visual_featuresT


class GraphReasoning(nn.Module):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
    """
    def __init__(self, sim_dim):
        super(GraphReasoning, self).__init__()
        self.percentage = 0.1
        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()
        self.init_weights()

    def forward(self, sim_emb):

        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.bmm(sim_query, sim_key.permute(0, 2, 1))

        # ######
        # batch_size, lengths = sim_edge.size(0), sim_edge.size(1)
        # count = math.ceil(lengths * self.percentage)
        # temp = (torch.ones(batch_size * lengths, lengths) * (-1e8)).cuda()
        # sim_edge = sim_edge.view(-1, lengths)
        # topk_value, topk_index = torch.topk(sim_edge, count, dim=-1, largest=False)
        # sim_edge = sim_edge.scatter_(-1, topk_index, temp).cuda()
        # sim_edge = sim_edge.view(batch_size, lengths, lengths)
        # #######

        sim_edge = torch.softmax(sim_edge, dim=-1)
        # sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr = torch.bmm(sim_edge, sim_emb)
        # sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr + sim_emb

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_state_dict(self, state_dict):

        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(GraphReasoning, self).load_state_dict(new_state)

class Reason_i2t(nn.Module):

    def __init__(self,
                 feat_dim,
                 hid_dim,
                 out_dim,
                 opt):
        '''
        ## Variables:
        - feat_dim: dimensionality of input image features
        - out_dim: dimensionality of the output
        - dropout: dropout probability
        '''

        super(Reason_i2t, self).__init__()
        # Set parameters
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.opt = opt
        self.sim_w = nn.utils.weight_norm(nn.Linear(opt.embed_size, opt.sim_dim))
        # reasoning matching scores
        self.graph_reasoning = nn.ModuleList([GraphReasoning(opt.sim_dim) for i in range(3)])
        self.activation = nn.Sigmoid()
        self.out_1 = nn.utils.weight_norm(nn.Linear(opt.sim_dim, hid_dim))
        self.out_2 = nn.utils.weight_norm(nn.Linear(hid_dim, out_dim))
    #     self.init_weights()
    #
    # def init_weights(self):
    #     for m in self.children():
    #         if isinstance(m, nn.Linear):
    #             r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
    #             m.weight.data.uniform_(-r, r)
    #             m.bias.data.fill_(0)

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(Reason_i2t, self).load_state_dict(new_state)

    def PairNorm(self, x):
        alpha = 10.0
        col_mean = x.mean(dim=0)
        x = x-col_mean
        rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
        x = alpha * x / rownorm_mean
        return x

    def i2t_scores(self, tnodes, vnodes):
        batch_size, num_regions = vnodes.size(0), vnodes.size(1)
        weighted_visual_features, _ = weighted_Features(tnodes, vnodes)
        sim_mvector = torch.pow(torch.sub(weighted_visual_features, vnodes), 2)
        sim_mvector = l2norm(self.sim_w(sim_mvector), dim=-1)
        # Reasoning
        for module in self.graph_reasoning:
            sim_mvector = module(sim_mvector)
        sim = self.out_1(sim_mvector.view(batch_size * num_regions, -1)).view(batch_size, num_regions, -1).tanh()
        sim = self.out_2(sim)
        sim = sim.view(batch_size, -1).mean(dim=1, keepdim=True)
        return sim, weighted_visual_features, _

    def forward(self, images, captions, cap_lens, opt):
        similarities = []
        visual_features = []
        textual_features = []
        n_image, n_caption = images.size(0), captions.size(0)
        for i in range(n_caption):
            n_word = cap_lens[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            i2t, weighted_visual_features, weighted_textual_features = self.i2t_scores(cap_i_expand, images)
            similarities.append(i2t)
            weighted_visual_features = weighted_visual_features.mean(dim=0, keepdim=True).squeeze(0)
            weighted_textual_features = weighted_textual_features.mean(dim=0, keepdim=True).squeeze(0)
            visual_features.append(weighted_visual_features.mean(dim=0, keepdim=True))
            textual_features.append(weighted_textual_features.mean(dim=0, keepdim=True))
        similarities = torch.cat(similarities, 1)
        visual_features = torch.cat(visual_features, 0)
        textual_features = torch.cat(textual_features, 0)
        return similarities, visual_features, textual_features


class Reason_t2i(nn.Module):

    def __init__(self,
                 feat_dim,
                 hid_dim,
                 out_dim,
                 opt):

        super(Reason_t2i, self).__init__()

        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.opt = opt
        self.sim_w = nn.utils.weight_norm(nn.Linear(opt.embed_size, opt.sim_dim))
        self.graph_reasoning = nn.ModuleList([GraphReasoning(opt.sim_dim) for i in range(3)])
        self.activation = nn.Sigmoid()
        self.out_1 = nn.utils.weight_norm(nn.Linear(opt.sim_dim, hid_dim))
        self.out_2 = nn.utils.weight_norm(nn.Linear(hid_dim, out_dim))
    #     self.init_weights()
    #
    # def init_weights(self):
    #
    #     for m in self.children():
    #         if isinstance(m, nn.Linear):
    #             r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
    #             m.weight.data.uniform_(-r, r)
    #             m.bias.data.fill_(0)

    def load_state_dict(self, state_dict):

        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(Reason_t2i, self).load_state_dict(new_state)

    def t2i_scores(self, vnodes, tnodes):
        batch_size, num_words = tnodes.size(0), tnodes.size(1)
        weighted_textual_features, _ = weighted_Features(vnodes, tnodes)
        sim_mvector = torch.pow(torch.sub(weighted_textual_features, tnodes), 2)
        sim_mvector = l2norm(self.sim_w(sim_mvector), dim=-1)
        # Reasoning
        for module in self.graph_reasoning:
            sim_mvector = module(sim_mvector)
        sim = self.out_1(sim_mvector.view(batch_size * num_words, -1)).view(batch_size, num_words, -1).tanh()
        sim = self.out_2(sim)
        sim = sim.view(batch_size, -1).mean(dim=1, keepdim=True)
        return sim, weighted_textual_features, _

    def forward(self, images, captions, cap_lens, opt):
        n_image = images.size(0)
        n_caption = captions.size(0)
        similarities = []
        textual_features = []
        visual_features = []
        for i in range(n_caption):
            # Get the i-th text description tanh()
            n_word = cap_lens[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            t2i, weighted_textual_features, weighted_visual_features = self.t2i_scores(images, cap_i_expand)
            similarities.append(t2i)
            weighted_textual_features = weighted_textual_features.mean(dim=0, keepdim=True).squeeze(0)
            weighted_visual_features = weighted_visual_features.mean(dim=0, keepdim=True).squeeze(0)
            textual_features.append(weighted_textual_features.mean(dim=0, keepdim=True))
            visual_features.append(weighted_visual_features.mean(dim=0, keepdim=True))
        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)
        textual_features = torch.cat(textual_features, 0)
        visual_features = torch.cat(visual_features, 0)
        return similarities, visual_features, textual_features
