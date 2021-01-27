import numpy as np
from torch.nn import functional as F
import torch
from model.fusions.fusions import MFB
import torch.nn as nn

class ConcatELUAttn(nn.Module):
    def __init__(self, module_dim=768):
        super(ConcatELUAttn, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2*module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):
        """
        Args:
            question_rep: [Tensor] (batch_size, module_dim)
            visual_feat: [Tensor] (batch_size, num_of_clips, module_dim)
        return:
            visual_distill representation [Tensor] (batch_size, module_dim)
        """
        visual_feat = self.dropout(visual_feat) # TODO
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)
        q_proj = q_proj.unsqueeze(1)

        v_q_cat = torch.cat((v_proj, q_proj.expand_as(v_proj) * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)

        return v_distill

class MFBAttn(nn.Module):
    def __init__(self, module_dim=768):
        super(MFBAttn, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = MFB([module_dim, module_dim], module_dim, mm_dim=module_dim, factor=2)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):
        """
        Args:
            question_rep: [Tensor] (batch_size, module_dim)
            visual_feat: [Tensor] (batch_size, num_of_clips, module_dim)
        return:
            visual_distill representation [Tensor] (batch_size, module_dim)
        """
        visual_feat = self.dropout(visual_feat) # TODO
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)
        q_proj = q_proj.unsqueeze(1)

        v_q_cat = self.cat(v_proj, q_proj.expand_as(v_proj) * v_proj)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)

        return v_distill

class MFBSimpleAttn(nn.Module):
    def __init__(self, module_dim=768):
        super(MFBAttn, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = MFB([2 * module_dim, module_dim * 2], module_dim, mm_dim=module_dim, factor=2)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):
        """
        Args:
            question_rep: [Tensor] (batch_size, module_dim)
            visual_feat: [Tensor] (batch_size, num_of_clips, module_dim)
        return:
            visual_distill representation [Tensor] (batch_size, module_dim)
        """
        visual_feat = self.dropout(visual_feat) # TODO
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)
        q_proj = q_proj.unsqueeze(1)

        v_q_cat = self.cat(v_proj, q_proj.expand_as(v_proj))

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)

        return v_distill

class SimpleConcatELUAttn(nn.Module):
    def __init__(self, module_dim=768):
        super(SimpleConcatELUAttn, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2*module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):
        """
        Args:
            question_rep: [Tensor] (batch_size, module_dim)
            visual_feat: [Tensor] (batch_size, num_of_clips, module_dim)
        return:
            visual_distill representation [Tensor] (batch_size, module_dim)
        """
        visual_feat = self.dropout(visual_feat) # TODO
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)
        q_proj = q_proj.unsqueeze(1)

        v_q_cat = torch.cat((v_proj, q_proj.expand_as(v_proj)), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)

        return v_distill

class ContextSelfAttn(nn.Module):
    def __init__(self, module_dim=768):
        super(ContextSelfAttn, self).__init__()
        self.module_dim = module_dim
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.attn = nn.Linear(module_dim, 1)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, visual_feat):
        """
        Args:
            question_rep: [Tensor] (batch_size, module_dim)
            visual_feat: [Tensor] (batch_size, num_of_clips, module_dim)
        return:
            visual_distill representation [Tensor] (batch_size, module_dim)
        """
        visual_feat = self.dropout(visual_feat) # TODO
        v_proj = self.v_proj(visual_feat)
        v_q_cat = self.activation(v_proj)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)

        return v_distill

class SimpleOutputUnitOpenEnded(nn.Module): # concat->Linear classifier
    def __init__(self, module_dim=512, num_answers=1000):
        super(SimpleOutputUnitOpenEnded, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.classifier(out)

        return out

class GateOutputUnitOpenEnded(nn.Module): # concat->Linear classifier
    def __init__(self, module_dim=512, num_answers=1000):
        super(GateOutputUnitOpenEnded, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))
        self.gate = nn.Linear(module_dim * 2, module_dim * 2)

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        gate = self.gate(out)
        out = gate * out
        out = self.classifier(out)

        return out

