import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys, math

# demo1: GCN
class GraphConvolution(nn.Module):
    '''
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    '''
    def __init__(self, in_features, out_features, bias = False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features # feat_dim
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features,out_features))
        if bias:
            self.bias = nn.Parameter(torch.tensor(1,1,out_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        '''
        :param input: num_nodes * feat_dim
        :param adj: num_nodes * num_nodes
        :return:
        '''
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def process_adj(A):
    # Get size of the adjacency matrix
    size = len(A)
    # Get the degrees for each node
    degrees = []
    for node_adjaceny in A:
        num = 0
        for node in node_adjaceny:
            if node == 1.0:
                num = num + 1
        # Add an extra for the "self loop"
        num = num + 1
        degrees.append(num)
    # Create diagonal matrix D from the degrees of the nodes
    D = np.diag(degrees)
    # Cholesky decomposition of D
    D = np.linalg.cholesky(D)
    # Inverse of the Cholesky decomposition of D
    D = np.linalg.inv(D)
    # Create an identity matrix of size x size
    I = np.eye(size)
    # Turn adjacency matrix into a numpy matrix
    A = np.matrix(A)
    # Create A hat
    A_hat = A + I
    # Return A_hat
    return torch.Tensor(A_hat), torch.Tensor(D) # the same as Variable
# D.mm(A).mm(D)

class PunishGraphAttentionLayer(nn.Module):
    '''
    Simple GAT layer, similar to https://arxiv,org/abs/1710.10903
    '''
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(PunishGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features # input_dim
        self.out_features = out_features # output_dim
        self.alpha = alpha
        self.concat = concat # True: ELU, or False: No ELU

        self.W = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        self.a = nn.Linear(2 * out_features, 1)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, h, adj, scores):
        Wh = self.W(h)  # (batch_size, num_of_nodes, output_dim)
        a_input = self._prepare_attentional_mechanism_input(Wh) # (batch_size, num_of_nodes, num_of_nodes, 2 * output_dim)
        e = self.leakyrelu(self.a(a_input).squeeze(-1)) # attention coefficients

        zero_vec = -9e15 * torch.ones_like(e) # -limit
        attention = torch.where(adj > 0, e, zero_vec) # scores
        if scores is not None:
            #print(Wh.shape, scores.shape)
            Wh = Wh * scores

        attention = F.softmax(attention, dim=-1) # (batch_size, num_of_nodes, num_of_nodes)
        attention = F.dropout(attention, self.dropout, training=self.training) # (batch_size, num_of_nodes, num_of_nodes)
        h_prime = torch.bmm(attention, Wh) # (batch_size, num_of_nodes, output_dim)

        if self.concat:
            return F.elu(h_prime) # common non-linearity
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[1]  # number of nodes
        bs = Wh.size()[0]

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat(1, 1, N).view(bs, N * N, self.out_features)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(bs, N, N, 2 * self.out_features)

#

class punishGAT(nn.Module): # n_hid: relatively small
    def __init__(self, n_feat, n_hid, dropout, alpha, n_heads, q_attn=True):
        """
        Multi-head version of GAT
        """
        super(punishGAT, self).__init__()
        self.dropout = dropout
        self.q_attn = q_attn
        # Multi-head
        self.attentions = [PunishGraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # add into pytorch modules_dict, can be indexed by the name
        # 输出层，也通过图注意力层来实现，可实现分类、预

    def forward(self, x, adj, scores):
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, adj, scores) for att in self.attentions], dim=2)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        return x

# demo-2: GAT
class GraphAttentionLayer(nn.Module):
    '''
    Simple GAT layer, similar to https://arxiv,org/abs/1710.10903
    '''
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features # input_dim
        self.out_features = out_features # output_dim
        self.alpha = alpha
        self.concat = concat # True: ELU, or False: No ELU

        self.W = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        self.a = nn.Linear(2 * out_features, 1)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = self.W(h)  # (batch_size, num_of_nodes, output_dim)
        a_input = self._prepare_attentional_mechanism_input(Wh) # (batch_size, num_of_nodes, num_of_nodes, 2 * output_dim)
        e = self.leakyrelu(self.a(a_input).squeeze(-1)) # attention coefficients

        zero_vec = -9e15 * torch.ones_like(e) # -limit
        attention = torch.where(adj > 0, e, zero_vec) # scores
        attention = F.softmax(attention, dim=-1) # (batch_size, num_of_nodes, num_of_nodes)
        attention = F.dropout(attention, self.dropout, training=self.training) # (batch_size, num_of_nodes, num_of_nodes)
        h_prime = torch.bmm(attention, Wh) # (batch_size, num_of_nodes, output_dim)

        if self.concat:
            return F.elu(h_prime) # common non-linearity
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[1]  # number of nodes
        bs = Wh.size()[0]

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat(1, 1, N).view(bs, N * N, self.out_features)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(bs, N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# demo-3: Multi-head GAT
class GAT(nn.Module): # n_hid: relatively small
    def __init__(self, n_feat, n_hid, dropout, alpha, n_heads):
        """
        Multi-head version of GAT
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        # Multi-head
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # add into pytorch modules_dict, can be indexed by the name
        # 输出层，也通过图注意力层来实现，可实现分类、预

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合

        return x



        # demo-4: KNN Graph
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair

def construct_graph(features,topk):
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i,:],-(topk+1))[-(topk+1):]
        inds.append(ind)

    A = np.zeros((dist.shape[0],dist.shape[0]))
    for i, v in enumerate(inds):
        for vv in v:
            A[i,vv] = 1
    return A

# demo-5: Gate_GINlayer-meaningless
class ginLayer(nn.Module):
    def __init__(self, input_dim, proj_dim=512, dropout=0.1, num_hop=3, batch_norm=False, num_rel=3):
        super(ginLayer, self).__init__()

        self.proj_dim = proj_dim
        self.num_hop = num_hop
        self.num_rel = num_rel
        self.epsilon = nn.Parameter(torch.Tensor([0])) # intialized with 0

        self.fa = nn.Sequential(nn.Linear(input_dim + proj_dim, proj_dim), nn.Dropout(dropout))

        for i in range(self.num_hop):
            for j in range(self.num_rel):
                setattr(self, "mlp{}{}".format(i+1, j+1), nn.Sequential(nn.Linear(input_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout)))

    def forward(self, input, input_mask, adj):
        # input: bs x max_nodes x node_dim
        # input_mask: bs x max_nodes
        # adj: bs x num_rel x max_nodes x max_nodes

        cur_input = input.clone()

        for i in range(self.num_hop):

            # replicate input
            multi_input = torch.stack([cur_input for i in range(self.num_rel)], dim=1) # bs x num_rel x max_nodes x node_dim

            # integrate neighbor information
            nb_output = torch.matmul(adj, multi_input) * input_mask.unsqueeze(-1).unsqueeze(1)  # bs x num_rel x max_nodes x node_dim

            # add cur node
            cur_update = (1 + self.epsilon)*multi_input + nb_output

            # apply different mlps for different relations
            update = torch.mean(torch.stack([getattr(self, "mlp{}{}".format(i+1, j+1))(cur_update[:,j,:,:].squeeze(1)) \
                                for j in range(self.num_rel)], dim=1), dim=1)* input_mask.unsqueeze(-1) # bs x max_nodes x node_dim

            # get gate values
            gate = torch.sigmoid(self.fa(torch.cat((update, cur_input), -1))) * input_mask.unsqueeze(
                -1)  # bs x max_node x node_dim

            # apply gate values
            cur_input = gate * update + (1 - gate) * cur_input  # bs x max_node x node_dim

        return cur_input

# demo-6: Gate_GATlayer--a different attention mechanism
class gatLayer(nn.Module):
    def __init__(self, input_dim, proj_dim=512, dropout=0.1, num_hop=3, num_rel=2, batch_norm=False):
        super(gatLayer, self).__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.num_hop = num_hop
        self.gcn_num_rel = num_rel # by default--2

        self.sfm = nn.Softmax(-1)

        for i in range(self.gcn_num_rel):
            setattr(self, "fr{}".format(i + 1), nn.Sequential(nn.Linear(input_dim, proj_dim), nn.Dropout(dropout)))

            # attention weight
            setattr(self, "fa{}".format(i + 1), nn.Linear(input_dim, input_dim, bias=False))

        self.fs = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.Dropout(dropout))

        self.fg = nn.Sequential(nn.Linear(input_dim + proj_dim, proj_dim), nn.Dropout(dropout))

    def forward(self, input, input_mask, adj):
        # input: bs x max_nodes x node_dim
        # input_mask: bs x max_nodes
        # adj: bs x 3 x max_nodes x max_nodes
        # num_layer: number of layers; note that the parameters of all layers are shared

        cur_input = input.clone()

        for i in range(self.num_hop): # 3

            # get attention
            att_list = []
            for j in range(self.gcn_num_rel): # 2
                att = torch.bmm(getattr(self, "fa{}".format(j + 1))(cur_input), # input_dim * input_dim + max_nodes * input_dim
                                cur_input.transpose(1, 2).contiguous()) / math.sqrt(
                    self.input_dim)  # bs x max_nodes x max_nodes
                zero_vec = -9e15 * torch.ones_like(att)
                att = torch.where(adj[:, j, :, :].squeeze(1) > 0, att, zero_vec)
                att_list.append(self.sfm(att))

            att_matrices = torch.stack(att_list, dim=1)

            # integrate neighbor information
            nb_output = torch.stack([getattr(self, "fr{}".format(i + 1))(cur_input) for i in range(self.gcn_num_rel)],
                                    1) * input_mask.unsqueeze(-1).unsqueeze(1)  # bs x 2 x max_nodes x node_dim

            # apply different types of connections, which are encoded in adj matrix
            update = torch.sum(torch.matmul(att_matrices, torch.matmul(adj, nb_output)), dim=1, keepdim=False) + \
                     self.fs(cur_input) * input_mask.unsqueeze(-1)  # bs x max_node x node_dim

            # get gate values
            gate = torch.sigmoid(self.fg(torch.cat((update, cur_input), -1))) * input_mask.unsqueeze(
                -1)  # bs x max_node x node_dim

            # apply gate values
            cur_input = gate * torch.tanh(update) + (1 - gate) * cur_input  # bs x max_node x node_dim

        return cur_input

# demo-7: Gated-GCN
# Entity-GCN, avoid overwriting past information-smoothing problem, all layers share parameters
class gcnLayer(nn.Module):
    def __init__(self, input_dim, proj_dim=512, dropout=0.1, num_hop=3, gcn_num_rel=2, batch_norm=False):
        super(gcnLayer, self).__init__()
        self.proj_dim = proj_dim
        self.num_hop = num_hop # num_of_layers
        self.gcn_num_rel = gcn_num_rel

        for i in range(gcn_num_rel):
            setattr(self, "fr{}".format(i+1), nn.Sequential(nn.Linear(input_dim, proj_dim), nn.Dropout(dropout, inplace=False)))

        self.fs = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.Dropout(dropout, inplace=False))

        self.fa = nn.Sequential(nn.Linear(input_dim + proj_dim, proj_dim), nn.Dropout(dropout, inplace=False))

    def forward(self, input, adj):
        # input: bs x max_nodes x node_dim
        # input_mask: bs x max_nodes
        # adj: bs x 3 x max_nodes x max_nodes
        # num_layer: number of layers; note that the parameters of all layers are shared

        cur_input = input.clone()

        for i in range(self.num_hop): # by default, 3
            # integrate neighbor information
            nb_output = torch.stack([getattr(self, "fr{}".format(i+1))(cur_input) for i in range(self.gcn_num_rel)], # bs x max_nodes x proj_dim
                                    1)# bs x 2 x max_nodes x node_dim

            # apply different types of connections, which are encoded in adj matrix
            update = torch.sum(torch.matmul(adj,nb_output), dim=1, keepdim=False) + \
                     self.fs(cur_input) # bs x max_node x node_dim

            # get gate values
            gate = torch.sigmoid(self.fa(torch.cat((update, cur_input), -1))) # bs x max_node x node_dim

            # apply gate values
            cur_input = gate * torch.tanh(update) + (1 - gate) * cur_input  # bs x max_node x node_dim

        return cur_input

# TODO: Question-gated information propogate
# demo-8: common-GCN
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        #self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        # 1.first layer: GCN+relu
        # 2.second layer: GCN+no relu
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        #x = self.gc2(x, adj)
        return x

# demo-9: SFGCN-bjupt
class AttentionSFGCN(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(AttentionSFGCN, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z) # num_of_nodes * 3 * 1
        beta = torch.softmax(w, dim=1) #
        return (beta * z).sum(1), beta

class SFGCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
        super(SFGCN, self).__init__()

        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = AttentionSFGCN(nhid2)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj, fadj):
        emb1 = self.SGCN1(x, sadj) # Special_GCN out1 -- sadj structure graph
        com1 = self.CGCN(x, sadj)  # Common_GCN out1 -- sadj structure graph
        com2 = self.CGCN(x, fadj)  # Common_GCN out2 -- fadj feature graph
        emb2 = self.SGCN2(x, fadj) # Special_GCN out2 -- fadj feature graph
        # all results are implemented without relu
        Xcom = (com1 + com2) / 2
        ##attention
        emb = torch.stack([emb1, emb2, Xcom], dim=1) # num_of_nodes * 3 * feat_dim
        emb, att = self.attention(emb)
        output = self.MLP(emb)
        return output, att, emb1, com1, com2, emb2, emb

