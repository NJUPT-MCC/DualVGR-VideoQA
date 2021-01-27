import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from model.init_weight import init_weight

class AttentionSFGCN(nn.Module):
    def __init__(self,in_size,hidden_size=16):
        super(AttentionSFGCN,self).__init__()
        self.project = nn.Sequential(
                nn.Linear(in_size,hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size,1,bias=False)
            )
    
    def forward(self,z):
        w = self.project(z)
        beta = torch.softmax(w,dim=1)
        return (beta*z).sum(1),beta

class ScaledDotProductAttention(nn.Module): #  8*bs, 20, 32 , 一般来说是在sequence_length上做self-attention
    ''' Scaled Dot-Product Attention '''
    # q vs k: scaled dot-product attention-v

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2)) # 8*bs, 20, 20
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn) # 还加个dropout
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module): # Transformer phase-1
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1): # 8, 256, 32, 32
        super().__init__()

        self.n_head = n_head # 8
        self.d_k = d_k # 32
        self.d_v = d_v # 32

        self.w_qs = nn.Linear(d_model, n_head * d_k) # 256, 256
        self.w_ks = nn.Linear(d_model, n_head * d_k) # 256, 256
        self.w_vs = nn.Linear(d_model, n_head * d_v) # 256, 256
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head # 32, 32, 8

        sz_b, len_q, _ = q.size() # _: feature, len_q: vgg_feature 20
        sz_b, len_k, _ = k.size() # bs, 20, 256
        sz_b, len_v, _ = v.size() #

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
# 2,0,1,3: 8, bs, 20, 32
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask) # output; 8*bs, 20, 32

        output = output.view(n_head, sz_b, len_q, d_v) # 8, bs, 20, 32
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
# 1,2,0,3: bs,20,8,32--->bs,20,256
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual) # v+q?

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1): # 256, 512
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout) # 8, 256, 32, 32
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, q, k, v, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            q, k, v, mask=slf_attn_mask)
        if non_pad_mask:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        if slf_attn_mask:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class Attention_C(nn.Module):
    def __init__(self, dim, num_hid, dropout=0.5):
        super(Attention_C, self).__init__()
        self.num_hid = num_hid
        head = 16
        self.head = head
        self.op = nn.Parameter(torch.zeros([1, head, 1]))
        nn.init.constant(self.op, 1)
        self.fc1 = nn.Linear(dim, num_hid)
        self.w = nn.Linear(head,1)
        self.d = dropout

    def forward(self, v, q1):
        batch = v.size(0)

        # op = self.op.expand([batch, 64, 1])

        q1_proj = self.fc1(q1).view(batch, -1, self.num_hid).expand([batch, self.head, self.num_hid])

        score = F.tanh(torch.bmm(self.op.expand([batch, self.head, 1]), v) + q1_proj)
        score = score.transpose(1, 2)
        weight = F.sigmoid(self.w(score)).transpose(1, 2)
        feature = weight * v
        return feature

class RNNEncoder(nn.Module): #这种定义RNN和LSTM的代码很值得我学习并且复用
    def __init__(self, word_size, hidden_size, bidirectional=True,
                 drop_prob=0, n_layers=2, rnn_type='lstm'):
        super(RNNEncoder, self).__init__()
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=drop_prob)
        self.num_dirs = 2 if bidirectional else 1

    def forward(self, embedded, input_lengths): # ques_emb, quesL
        """
        Inputs:
        - input_labels: long (batch, seq_len)-input_length
        embedded: bs * max_sentence_length*embeddingsize
        input_lengths: bs * 1（每个sentence真实的长度）
        Outputs:
        - output  : float (batch, max_len, hidden_size * num_dirs)
        - hidden  : float (batch, num_layers * num_dirs * hidden_size)
        - embedded: float (batch, max_len, word_vec_size)
        """
        # make ixs
        batch_size = input_lengths.size(0)
        mask0 = input_lengths.eq(0) # 完全不存在的句子怎么办呢？
        input_lengths.masked_fill_(mask0, 1) #在mask处value为1的值进行填充，用1来填充input_lengths
        input_lengths_list = input_lengths.data.cpu().numpy().tolist() #list形式的input length，处理了oov

        sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist()  # list of sorted input_lengths
        sort_ixs = np.argsort(input_lengths_list)[::-1].tolist()  # list of int sort_ixs, descending

        s2r = {s: r for r, s in enumerate(sort_ixs)}  # O(n)
        recover_ixs = [s2r[s] for s in range(batch_size)]  # list of int recover ixs---按照batchsize的第几个句子对应的是哪一个序号-按长度逆序

        # sort input_labels by descending order
        embedded = embedded[sort_ixs] #一个batch里的句子重新按照逆序来排列一下

        # embed
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True) # pack之后，原来填充的 PAD（一般初始化为0）占位符被删掉了。
# 变长序列压紧对于精度也是很重要的
        # forward rnn
        self.rnn.flatten_parameters() # 类似contiguous，使得数据是连续进行排列的
        output, hidden = self.rnn(embedded)  # output-seq_len * batch_size * hidden_size * num_directions
#output就是所有hidden state，hidden应该是最后一个
        # recover
        # embedded (batch, seq_len, word_vec_size)
        embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)
        embedded = embedded[recover_ixs]

        # recover rnn
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # (batch, max_len, hidden)
        output = output[recover_ixs] #按照原来batch的顺序再搞一下

        # recover hidden
        if self.rnn_type == 'lstm':
            hidden = hidden[0]  # we only use hidden states for the final hidden representation
        hidden = hidden[:, recover_ixs, :]  # (num_layers * num_dirs, batch, hidden_size)
        hidden = hidden.transpose(0, 1).contiguous()  # (batch, num_layers * num_dirs, hidden_size)
        hidden = hidden.view(hidden.size(0), -1)  # (batch, num_layers * num_dirs * hidden_size)

        return output, hidden, embedded

class TanhAttention(nn.Module):
    def __init__(self, d_model, dropout=0.0, direction=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ws1 = nn.Linear(d_model, d_model, bias=True)
        self.ws2 = nn.Linear(d_model, d_model, bias=True)
        self.wst = nn.Linear(d_model, 1, bias=True)
        self.direction = direction

    def forward(self, x, memory, memory_mask=None):
        
        item1 = self.ws1(x)  # [nb, len1, d]
        item2 = self.ws2(memory)  # [nb, len2, d]
        item = item1.unsqueeze(2) + item2.unsqueeze(1)  # [nb, len1, len2, d]
        S = self.wst(torch.tanh(item)).squeeze(-1)  # [nb, len1, len2]
        if memory_mask is not None:
            memory_mask = memory_mask.unsqueeze(1)  # [nb, 1, len2]
            S = S.masked_fill(memory_mask == 0, -1e30)
            # for forward, backward, S: [nb, len, len]
            if self.direction == 'forward':
                length = S.size(1) # len1
                forward_mask = torch.ones(length, length) # len1 * len1
                for i in range(1, length):
                    forward_mask[i, 0:i] = 0
                S = S.masked_fill(forward_mask.cuda().unsqueeze(0) == 0, -1e30)
            elif self.direction == 'backward':
                length = S.size(1)
                backward_mask = torch.zeros(length, length)
                for i in range(0, length):
                    backward_mask[i, 0:i + 1] = 1
                S = S.masked_fill(backward_mask.cuda().unsqueeze(0) == 0, -1e30)
        S = self.dropout(F.softmax(S, -1))
        return torch.matmul(S, memory)  # [nb, len1, d]


class WordAttention(nn.Module): # sentence-weighted attn: word

    def __init__(self, input_dim):
        super(WordAttention, self).__init__()
        # initialize pivot
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, context, embedded, input_labels):
        """
        Inputs:
        - context : float (batch, seq_len, input_dim)
        - embedded: float (batch, seq_len, word_vec_size)
        - input_labels: long (batch, seq_len)
        Outputs:
        - attn    : float (batch, seq_len)
        - weighted_emb: float (batch, word_vec_size)
        """
        cxt_scores = self.fc(context).squeeze(2)  # (batch, seq_len)
        attn = F.softmax(cxt_scores, dim=1)  # (batch, seq_len), attn.sum(1) = 1.

        # mask zeros
        is_not_zero = (input_labels != 0).float()  # (batch, seq_len)
        attn = attn * is_not_zero  # (batch, seq_len)
        attn = attn / (attn.sum(1) + 1e-5).view(attn.size(0), 1).expand(attn.size(0), attn.size(1))  # (batch, seq_len)

        # compute weighted embedding
        attn3 = attn.unsqueeze(1)  # (batch, 1, seq_len)
        weighted_emb = torch.bmm(attn3, embedded)  # (batch, 1, word_vec_size)
        weighted_emb = weighted_emb.squeeze(1)  # (batch, word_vec_size)

        return attn, weighted_emb

# 1-dim: FC-tanh Self-Attention: Seq->vector
class SimpleSelfAttention(nn.Module): # fctanh+simpleselfatt
    def __init__(self, in_dim, inner_dim, dropout=0.3): # 256, 256
        super(SimpleSelfAttention, self).__init__()
        self.fc1 = FCNet([in_dim, inner_dim])
        self.w = nn.Linear(inner_dim, 1)
        self.w.apply(init_weight)
        self.d = dropout

    def forward(self, q, ques, mask=True): #why these FC layers?
        '''
        :param q: b,len,dim---来自rnn
        ques->序号word b,len
        :param mask:
        :return:
        '''
        batch, qlen, _ = q.size() #maxlength
        q_proj = self.fc1(q).view(batch, qlen, -1) # 256， 256-2 layer
        score = F.tanh(q_proj) #又是tanh的非线性变换
        score = self.w(score).view(batch, qlen) # bs, len, 256->bs, len, 1
        if mask:
            label = ques[:, :qlen]
            is_not_zero = label.eq(0)
            score.masked_fill_(is_not_zero, -np.inf) #在is_not_zero的位置用-inf填充
        weight = F.softmax(score.view(-1, 1, qlen), dim=2)
        if self.d>0:
            weight = F.dropout(weight, p=self.d, training=self.training)
        output = torch.bmm(weight, q)
        return output

class Gated_NLT(nn.Module):
    def __init__(self, in_dim, inner_dim, dropout=0.):
        super(Gated_NLT, self).__init__()
        self.fc1 = FCNet([in_dim, inner_dim])
        self.fc2 = FCNet([in_dim, inner_dim])
        self.d = dropout

    def forward(self, x):

        y = F.tanh(self.fc1(x))
        g = F.tanh(self.fc2(x))
        y = g*y
        return y
# self updated
class ScaledDotProductAttentionSFGCN(nn.Module): #  8*bs, 20, 32 , 一般来说是在sequence_length上做self-attention
    ''' Scaled Dot-Product Attention '''
    # q vs k: scaled dot-product attention-v

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.visual_appearance_proj = nn.Linear()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2)) # 8*bs, 20, 20
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn) # 还加个dropout
        output = torch.bmm(attn, v)

        return output, attn

class CoAttention(nn.Module): # scaled dot-product,no-mask
    def __init__(self,seq_len=16, input_dim=512, hidden_dim=512):
        super(CoAttention,self).__init__()
        self.seq_len = seq_len
        self.feat_dim = input_dim
        self.query_proj = nn.Linear(input_dim,hidden_dim)
        self.context_proj = nn.Linear(input_dim,hidden_dim)
        if adapt_scale: # Adaptation-Scale
            self.scale = nn.Parameter(torch.FloatTensor([1. / math.sqrt(input_dim)]))
        else:
            self.scale = 1. / math.sqrt(input_dim)

    def forward(self,query,context):
        # query: batch_size * seq_len * feat_dim
        # context: batch_size * seq_len * feat_dim
        q_proj = self.query_proj(query)
        c_proj = self.context_proj(context)
        c_proj = torch.transpose(c_proj,1,2)
        Attn_mat = torch.bmm(q_proj,c_proj)*self.scale # bs * seq_len * seq_len
        # updated context vector
        c_attn_softmax = F.softmax(Attn_mat,dim=2)
        c_coattn = torch.bmm(c_attn_softmax,c_proj)
        # updated query vector
        q_attn_softmax = F.softmax(Attn_mat,dim=1)
        q_attn_softmax = torch.transpose(q_attn_softmax,1,2)
        q_coattn = torch.bmm(q_attn_softmax,q_proj)

        return c_coattn, q_coattn
'''
class CoAttentionTanh(nn.Module): # scaled dot-product,no-mask
    def __init__(self,seq_len=16, input_dim=512, hidden_dim=512):
        super(CoAttention,self).__init__()
        self.seq_len = seq_len
        self.feat_dim = input_dim
        self.query_proj = nn.Linear(input_dim,hidden_dim)
        self.context_proj = nn.Linear(input_dim,hidden_dim)

    def forward(self,query,context):
        # query: batch_size * seq_len * feat_dim
        # context: batch_size * seq_len * feat_dim
        q_proj = self.query_proj(query)
        c_proj = self.context_proj(context)
        c_proj = torch.transpose(c_proj,1,2)
        Attn_mat =
        # updated context vector
        c_attn_softmax = F.softmax(Attn_mat,dim=2)
        c_coattn = torch.bmm(c_attn_softmax,c_proj)
        # updated query vector
        q_attn_softmax = F.softmax(Attn_mat,dim=1)
        q_attn_softmax = torch.transpose(q_attn_softmax,1,2)
        q_coattn = torch.bmm(q_attn_softmax,q_proj)

        return c_coattn, q_coattn
'''
class MultiHeadCoAttention(nn.Module):
    def __init__(self, n_head = 8, dq = 64, dk = 64, seq_len=16, input_dim=512):
        super(CoAttention, self).__init__()
        self.seq_len = seq_len
        self.n_head = n_head
        self.d_q = dq# input_dim // n_head
        self.d_k = dk# input_dim // n_head
        self.feat_dim = input_dim


        self.query_proj = nn.Linear(input_dim, n_head * dq)
        self.context_proj = nn.Linear(input_dim, n_head * dk)
        if adapt_scale:  # Adaptative-Scale
            self.scale = nn.Parameter(torch.FloatTensor([1. / math.sqrt(input_dim)]))
        else:
            self.scale = 1. / math.sqrt(input_dim)

    def forward(self, query, context):
        # query: batch_size * seq_len * feat_dim
        # context: batch_size * seq_len * feat_dim
        bs, seq_len, input_dim = query.size()
        d_q, d_k, n_head = self.d_q, self.d_k, self.n_head
        q_proj = self.query_proj(query).view(bs, seq_len, n_head, d_q)
        c_proj = self.context_proj(context).view(bs, seq_len, n_head, d_k)
        q_proj = q_proj.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_q)  # (n_head * bs) x seq_len x dq
        c_proj = c_proj.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n_head * bs) x seq_len x dk
        # Multi-head Attention
        c_proj = torch.transpose(c_proj, 1, 2)
        Attn_mat = torch.bmm(q_proj, c_proj) * self.scale  # bs * seq_len * seq_len
        # updated context vector
        c_attn_softmax = F.softmax(Attn_mat, dim=2)
        c_coattn = torch.bmm(c_attn_softmax, c_proj)
        c_coattn = c_coattn.view(n_head, bs, seq_len, d_k)
        c_coattn = c_coattn.permute(1, 2, 0, 3).contiguous().view(bs, seq_len, -1)
        # updated query vector
        q_attn_softmax = F.softmax(Attn_mat, dim=1)
        q_attn_softmax = torch.transpose(q_attn_softmax, 1, 2)
        q_coattn = torch.bmm(q_attn_softmax, q_proj)
        q_coattn = q_coattn.view(n_head, bs, seq_len, d_k)
        q_coattn = q_coattn.permute(1, 2, 0, 3).contiguous().view(bs, seq_len, -1)

        return c_coattn, q_coattn

def diversity_loss(attention): # FIXME TODO
    attention_t = torch.transpose(attention, 1, 2)
    num_features = attention.shape[1]
    res = torch.matmul(attention_t.view(-1, args.num_filters, num_features), attention.view(-1, num_features, args.num_filters)) - torch.eye(args.num_filters).cuda()
    res = res.view(-1, args.num_filters*args.num_filters)
    return torch.norm(res, p=2, dim=1).sum() / attention.size(0)









