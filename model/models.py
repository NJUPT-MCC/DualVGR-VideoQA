import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys, math
from .Preprocessing import *
from .AnswerDecoder import *
from .init_weight import init_weight
from model.fusions.fusions import MFB
from model.GraphNN import *
import scipy.sparse as sp
from .utils import *
from .Attention import *

#from .position import *

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices,values,shape)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum,-1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class DualVGR(nn.Module):
    def __init__(self, vision_dim=2048, module_dim=768, word_dim=300, vocab=None, num_of_nodes=8, graph_module='GCN', graph_layers=1,unit_layers=2):
        super(DualVGR, self).__init__()
        self.feature_aggregation = ContextSelfAttn(module_dim)

        encoder_vocab_size = len(vocab['question_token_to_idx'])  # question_vocabsize
        self.num_classes = len(vocab['answer_token_to_idx'])  # ans_vocabsize
        self.linguistic_input_unit = InputUnitLinguisticDynamic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                         rnn_dim=module_dim, textual_encoder='LSTM')
        self.visual_appearance_input_unit = VisualAppearanceEncoder(appearance_dim=vision_dim, module_dim=module_dim,
                                                                    bidirectional=True)
        self.visual_motion_input_unit = nn.Linear(vision_dim, module_dim)

        self.visual_input_unit = DualVGRUnit_multiple(word_dim=word_dim, module_dim=module_dim, num_of_nodes=num_of_nodes, appearance_graph_layers=graph_layers, motion_graph_layers=graph_layers, graph_module=graph_module, unit_layers=unit_layers)

        self.output_unit = SimpleOutputUnitOpenEnded(module_dim=module_dim, num_answers=self.num_classes)

        init_modules(self.modules(), w_init="xavier_uniform")
        nn.init.uniform_(self.linguistic_input_unit.encoder_embed.weight, -1.0, 1.0)

    def forward(self, video_appearance_feat, video_motion_feat, question,
                question_len):
        """
        Args:
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question.size(0)
        # get image, word, and sentence embeddings
        question_embedding, word_embedding, dynamic_question_embedding = self.linguistic_input_unit(question,
                                                                                                    question_len)
        # question_embedding: (batch_size, module_dim)
        # word_embedding: (batch_size, seq_len, module_dim) @ word features
        # dynamic_question_embedding: (batch_size, seq_len, module_dim)
        video_appearance_feat = self.visual_appearance_input_unit(video_appearance_feat)  # appearance feature-LSTM: [Tensor] batch_size * num_of_clips * module_dim
        video_motion_feat = self.visual_motion_input_unit(video_motion_feat)  # motion feature-Linear: [Tensor] batch_size * num_of_clips * module_dim
        # Reasoning module
        ## 1. layer-1
        visual_embedding, aq_embed, mq_embed, com_app, com_motion, aq_fusion, mq_fusion = self.visual_input_unit(
            video_appearance_feat, video_motion_feat, dynamic_question_embedding, word_embedding, question_len)
        # output module
        visual_embedding = self.feature_aggregation(visual_embedding)
        out = self.output_unit(question_embedding, visual_embedding)

        return out, aq_embed, mq_embed, com_app, com_motion, aq_fusion, mq_fusion


class DualVGRUnit_multiple(nn.Module):  # 16, 8, 1, 2048, 512
    def __init__(self, word_dim=300, module_dim=512, num_of_nodes=8, appearance_graph_layers=1, motion_graph_layers=1, graph_module='GAT', unit_layers=3):
        super(DualVGRUnit_multiple, self).__init__()
        self.layers = unit_layers

        self.queryAttn = nn.ModuleList([QueryAttn(module_dim=module_dim) for _ in range(unit_layers)])
        self.queryPunish_appear = nn.ModuleList([QueryPunish(word_dim=word_dim, module_dim=module_dim) for _ in range(unit_layers)])
        self.queryPunish_motion = nn.ModuleList([QueryPunish(word_dim=word_dim, module_dim=module_dim) for _ in range(unit_layers)])
        if graph_module == 'GAT':
            self.appearance_GCN = nn.ModuleList([punishGAT(module_dim, module_dim // 4, dropout=0.15, alpha=0.01,
                                      n_heads=4) for _ in range(unit_layers * appearance_graph_layers)])  # n_feat, n_hid, dropout, alpha, n_heads
            self.motion_GCN = nn.ModuleList([punishGAT(module_dim, module_dim // 4, dropout=0.15, alpha=0.01, n_heads=4) for _ in range(unit_layers * motion_graph_layers)])
            self.acGCN = nn.ModuleList([punishGAT(module_dim, module_dim // 4, dropout=0.15, alpha=0.01,
                              n_heads=4) for _ in range(unit_layers * appearance_graph_layers)])  # n_feat, n_hid, dropout, alpha, n_heads
            self.mcGCN = nn.ModuleList([punishGAT(module_dim, module_dim // 4, dropout=0.15, alpha=0.01, n_heads=4) for _ in range(unit_layers * motion_graph_layers)])

        # Visual Fusion@Graph
        self.attention_appearance = nn.ModuleList([AttentionSFGCN(module_dim, module_dim) for _ in range(unit_layers)])
        self.attention_motion = nn.ModuleList([AttentionSFGCN(module_dim, module_dim) for _ in range(unit_layers)])
        # number of Graph Layers
        self.num_of_appearance_graph_layers = appearance_graph_layers
        self.num_of_motion_graph_layers = motion_graph_layers
        # visual feature fusion: appearance+motion
        self.visualfusion = MFB([module_dim, module_dim], module_dim)

        self.module_dim = module_dim
        self.activation = nn.ELU()

        adj = sp.coo_matrix(np.ones((num_of_nodes, num_of_nodes)))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))

        self.appearance_adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense().to('cuda:1')
        self.motion_adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense().to('cuda:1')

    def forward(self, appearance_video_feat, motion_video_feat, dynamic_question_embedding, word_embedding, question_len):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, vision_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, vision_dim)
            dynamic_question_embedding: [Tensor] (batch_size, seq_len, module_dim)
            word_embedding: [Tensor] (batch_size, seq_len, word_dim)
            question_len: [Tensor] (batch_size)
        return:
            encoded appearance feature: [Tensor] (batch_size, num_of_clips, module_dim)
            encoded motion feature: [Tensor] (batch_size, num_of_clips, module_dim)
        """
        batch_size = appearance_video_feat.size(0)
        appearance_adj = self.appearance_adj
        motion_adj = self.motion_adj
        aq_fusion_list = []
        mq_fusion_list = []
        com_app_list = []
        com_motion_list = []
 
        for i in range(self.layers):
            aq_fusion = appearance_video_feat
            mq_fusion = motion_video_feat
            # in each step, query representation should be computed through attention
            cycle_question_embedding,attn_score = self.queryAttn[i](word_embedding, dynamic_question_embedding, question_len) # (batch_size, word_dim)
            cycle_appearance_scores = self.queryPunish_appear[i](cycle_question_embedding, aq_fusion)
            cycle_motion_scores = self.queryPunish_motion[i](cycle_question_embedding, mq_fusion) # (batch_size, num_of_clips, module_dim)

            # graph reasoning module
            for j in range(self.num_of_appearance_graph_layers):
                com_app = self.acGCN[i+j](aq_fusion, appearance_adj, cycle_appearance_scores)
                aq_fusion = self.appearance_GCN[i+j](aq_fusion, appearance_adj, cycle_appearance_scores)
                aq_fusion_list.append(aq_fusion.cpu())
                com_app_list.append(com_app.cpu())
            
            for j in range(self.num_of_motion_graph_layers):
                com_motion = self.mcGCN[i+j](mq_fusion, motion_adj, cycle_motion_scores)
                mq_fusion = self.motion_GCN[i+j](mq_fusion, motion_adj, cycle_motion_scores)
                mq_fusion_list.append(mq_fusion.cpu())
                com_motion_list.append(com_motion.cpu())

            # Common vs Specific Fusion
            aq_embed = torch.stack([com_app, aq_fusion], dim=1)  # (batch_size, 2, num_of_clips, module_dim)
            mq_embed = torch.stack([com_motion, mq_fusion], dim=1)
            aq_embed, _ = self.attention_appearance[i](aq_embed)  # (batch_size, num_of_clips, module_dim)
            mq_embed, _ = self.attention_motion[i](mq_embed)
            # updated a,m
            appearance_video_feat = appearance_video_feat + aq_embed
            motion_video_feat = motion_video_feat + mq_embed
        # final visual feature representation
        visual_feature = self.visualfusion([appearance_video_feat, motion_video_feat])

        return visual_feature, aq_embed, mq_embed, com_app_list, com_motion_list, aq_fusion_list, mq_fusion_list

