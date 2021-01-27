from torch.nn import init
import torch
import torch.nn as nn
from .Attention import *
from model.fusions.fusions import MFB 
import torch.nn.functional as F

def init_modules(modules, w_init='kaiming_uniform'): # never have some negative effects on GCN
    if w_init == "normal":
        _init = init.normal_
    elif w_init == "xavier_normal":
        _init = init.xavier_normal_
    elif w_init == "xavier_uniform":
        _init = init.xavier_uniform_
    elif w_init == "kaiming_normal":
        _init = init.kaiming_normal_
    elif w_init == "kaiming_uniform":
        _init = init.kaiming_uniform_
    elif w_init == "orthogonal":
        _init = init.orthogonal_
    else:
        raise NotImplementedError
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            _init(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name:
                    _init(param)
# PCA Implementation
def meanX(dataX):
    return np.mean(dataX, axis=0)

def pca(XMat, k):
    average = meanX(XMat)
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)
    featValue, featVec = np.linalg.eig(covX)
    index = np.argsort(-featValue)
    finalData = []
    if k > n:
        print("k must lower than feature number")
        return
    else:
        selectVec = np.matrix(featVec.T[index[:k]])
        finalData = np.dot(XMat, selectVec.T)
    return finalData

def L2norm(x):
    print(x.shape)
    return x / torch.norm(x, p=2, dim=2) # batch_size * seq_len * module_dim

class QueryAttn(nn.Module):
    def __init__(self, module_dim = 768):
        super(QueryAttn, self).__init__()
        self.feat_enhance = nn.Linear(module_dim, module_dim)
        self.fc = nn.Linear(module_dim, 1)

    def forward(self, word_embedding, dynamic_question_embedding, question_len): # dynamic: [Tensor] batch_size * seq_len * module_dim
        bs = question_len.size(0)
        dynamic_question_embedding = F.normalize(self.feat_enhance(dynamic_question_embedding),p=2,dim=-1)
        attn = F.softmax(self.fc(dynamic_question_embedding).squeeze(2), dim=1) # batch_size * seq_len
        # mask zeros@words
        max_seq_len = dynamic_question_embedding.size(1)
        word_mask = torch.zeros((bs, max_seq_len)).to('cuda:1')
        for i in range(bs):
            temp_question_len = question_len[i]
            word_mask[i][:temp_question_len] = 1
        attn = attn * word_mask  # (batch, seq_len)
        attn = attn / (attn.sum(1) + 1e-5).view(bs, 1).expand(attn.size(0), attn.size(1))  # (batch, seq_len)
        # word embedding
        temp_attn = attn.unsqueeze(1) # (batch_size, 1, seq_len)
        #print(temp_attn.shape, word_embedding.shape)
        word_embedding = torch.bmm(temp_attn, word_embedding)
        word_embedding = word_embedding.squeeze(1) # (batch_size, word_dim)

        return word_embedding,attn

class QueryPunish(nn.Module):
    def __init__(self, word_dim=300, module_dim=768):
        super(QueryPunish, self).__init__()
        self.temp = np.sqrt(word_dim * module_dim)
        self.query_weight = nn.Linear(word_dim, module_dim)

    def forward(self, question_guided, visual_feature):
        """
        Inputs:
        - question_guided: [Tensor] (batch_size, word_dim)
        - visual_feature: [Tensor] (batch_size, num_of_clips, module_dim)
        Outputs:

        """
        query = self.query_weight(question_guided) # (batch_size, module_dim)
        query_scores = torch.bmm(visual_feature, query.unsqueeze(2)) # (batch_size, num_of_clips, 1)
        query_scores = torch.sigmoid(query_scores)
        query_scores = query_scores.expand(query_scores.size(0), query_scores.size(1), visual_feature.size(2)//4)
        ## visual_feature = query_scores * visual_feature # visual_feature
        return query_scores

class VisualEnhanceByQuery(nn.Module):
    def __init__(self, module_dim = 768):
        super(VisualEnhanceByQuery, self).__init__()
        self.t2v = TanhAttention(module_dim)
        self.gate1 = nn.Linear(module_dim, module_dim, bias = False)
        self.gate2 = nn.Linear(module_dim, module_dim, bias = False)
        self.tv_fusion = MFB([module_dim, module_dim], module_dim)

    def forward(self, dynamic_question_embedding, visual_embedding):
        """
        -Args
        dynamic_question_embedding: [Tensor] batch_size * seq_len * module_dim
        visual_embedding: [Tensor] batch_size * num_of_clips * module_dim
        """
        textual2visual = self.t2v(visual_embedding, dynamic_question_embedding)
        text_gate = torch.sigmoid(self.gate1(textual2visual))
        visual_final = text_gate * visual_embedding
        visual_gate = torch.sigmoid(self.gate2(visual_embedding))
        text_final = visual_gate * textual2visual
        final_fusion = self.tv_fusion([text_final, visual_final])
        return final_fusion





























