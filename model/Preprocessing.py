import numpy as np
from torch.nn import functional as F
from .utils import *
import torch
from .Attention import RNNEncoder, MultiHeadAttention

class DynamicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False, rnn_encoder = 'GRU'):
        super().__init__()
        self.batch_first = batch_first
        self.rnn = getattr(nn, rnn_encoder)(input_size, hidden_size, num_layers=num_layers, bias=bias,
                          batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.rnn.flatten_parameters()
        self.bidirectional = bidirectional

    def forward(self, x, seq_len, max_num_frames): # x: (batch_size, length, module_dim), seq_len: (batch_size), max_num_frames: scalar
        sorted_seq_len, sorted_idx = torch.sort(seq_len, dim=0, descending=True) # 都是batch_size 维度的张量
        _, original_idx = torch.sort(sorted_idx, dim=0, descending=False) #
        if self.batch_first:
            sorted_x = x.index_select(0, sorted_idx)
        else:
            # print(sorted_idx)
            sorted_x = x.index_select(1, sorted_idx)
        # 2nd-sorted input
        packed_x = nn.utils.rnn.pack_padded_sequence(
            sorted_x, sorted_seq_len.cpu().data.numpy(), batch_first=self.batch_first)

        out, (state, _) = self.rnn(packed_x)

        unpacked_x, unpacked_len = nn.utils.rnn.pad_packed_sequence(out, batch_first=self.batch_first)

        if self.batch_first:
            out = unpacked_x.index_select(0, original_idx)
            if out.shape[1] < max_num_frames: # batch_size * seq_len * feat_dim
                out = F.pad(out, [0, 0, 0, max_num_frames - out.shape[1]])
        else:
            out = unpacked_x.index_select(1, original_idx)
            if out.shape[0] < max_num_frames:
                out = F.pad(out, [0, 0, 0, 0, 0, max_num_frames - out.shape[0]])

        # state = state.transpose(0, 1).contiguous().view(out.size(0), -1)
        if self.bidirectional:
            state = torch.cat([state[0], state[1]], -1)
        return out,state # (batch_size, seq_len, feat_dim)

class InputUnitLinguistic(nn.Module): # BILSTM
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, bidirectional=True, textual_encoder='LSTM'):
        super(InputUnitLinguistic, self).__init__()

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2 # 256

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.concatRNN = getattr(nn, textual_encoder)(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
            #DynamicRNN(wordvec_dim, rnn_dim, num_layers=1, bias=True,
            #     batch_first=True, dropout=0.15, bidirectional=bidirectional, rnn_encoder = 'LSTM')
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.final_dropout = nn.Dropout(0.18)

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        max_ques_len = questions.size(1)
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        words = self.tanh(self.embedding_dropout(questions_embedding)) # tanh+dropout
        #question_embedding, output_embedding = self.concatRNN(words, question_len, max_ques_len)
        #question_embedding = F.dropout(question_embedding, self.dropout,
        #                               self.training)  # batch_size * max_seq_len * feat_dim
        #output_embedding = F.dropout(output_embedding, self.dropout, self.training)  # batch_size * feat_dim
        embed = nn.utils.rnn.pack_padded_sequence(words, question_len, batch_first=True, enforce_sorted=False)
        self.concatRNN.flatten_parameters()
        out, (question_embedding, _) = self.concatRNN(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]],-1)
        question_embedding = self.final_dropout(question_embedding)
        output_embedding, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        return question_embedding, words, output_embedding


class InputUnitLinguisticDynamic(nn.Module):  # BILSTM
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, bidirectional=True, textual_encoder='LSTM'):
        super(InputUnitLinguisticDynamic, self).__init__()

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2  # 256

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.concatRNN = DynamicRNN(wordvec_dim, rnn_dim, num_layers=1, bias=True,
             batch_first=True, dropout=0.15, bidirectional=bidirectional, rnn_encoder = textual_encoder)
        self.encoder = getattr(nn, textual_encoder)(wordvec_dim, rnn_dim, batch_first=True,
                                                      bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.final_dropout = nn.Dropout(0.18)

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        max_ques_len = questions.size(1)
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        words = self.tanh(self.embedding_dropout(questions_embedding))  # tanh+dropout
        output_embedding, _ = self.concatRNN(words, question_len, max_ques_len)

        embed = nn.utils.rnn.pack_padded_sequence(words, question_len, batch_first=True, enforce_sorted=False)
        self.encoder.flatten_parameters()
        out, (question_embedding, _) = self.encoder(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.final_dropout(question_embedding)
        #output_embedding, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=self.batch_first)

        return question_embedding, words, output_embedding

class MultiGranularInputUnitLinguistic(nn.Module): # BILSTM
    def __init__(self, vocab_size = None, wordvec_dim=300, rnn_dim=512, module_dim=512, rnn_encoder = 'LSTM', self_attn = False, bidirectional=True, n_heads = 4, dropout=0.15):
        super(MultiGranularInputUnitLinguistic, self).__init__()

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2 # 256

        self.dropout = dropout
        self.self_attn = self_attn

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()

        self.unigram_conv = nn.Conv1d(wordvec_dim, module_dim, 1, stride=1, padding = 0)
        self.bigram_conv = nn.Conv1d(wordvec_dim, module_dim, 2, stride=1, padding=1, dilation=2)
        self.trigram_conv = nn.Conv1d(wordvec_dim, module_dim, 3, stride=1, padding=2, dilation=2) # 1和1不是也可以嘛
        self.max_pool = nn.MaxPool2d((3,1))

        self.encoder = getattr(nn, rnn_encoder)(module_dim, rnn_dim, batch_first=True, bidirectional=bidirectional) # sentence encoder

        self.concatRNN = DynamicRNN(module_dim * 2 + wordvec_dim, module_dim, num_layers=1, bias=True,
                 batch_first=True, dropout=dropout, bidirectional=bidirectional, rnn_encoder = 'LSTM')

        if self.self_attn == True:
            self.self_attention = MultiHeadAttention(n_heads, module_dim * 2 + wordvec_dim, (module_dim * 2 + wordvec_dim)//n_heads, (module_dim * 2 + wordvec_dim)//n_heads, dropout)



    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        max_ques_len = questions.size(1)
        # 1st level: word-level features
        words = self.encoder_embed(questions)  # (batch_size, seq_len, word2vec_dim)
        words = self.tanh(F.dropout(words, self.dropout, self.training)) # (batch_size, seq_len, word2vec_dim)
        # 2nd level: phrase-level features
        words = words.permute(0,2,1) # (batch_size, word2vec_dim, seq_len)
        unigrams = torch.unsqueeze(self.unigram_conv(words), 2) # (batch_size, module_dim, 1, seq_len)
        bigrams = torch.unsqueeze(self.bigram_conv(words), 2) # (batch_size, module_dim, 1, seq_len)
        trigrams = torch.unsqueeze(self.trigram_conv(words), 2) # (batch_size, module_dim, 1, seq_len)
        words = words.permute(0,2,1) # (batch_size, module_dim, seq_len)
        phrase = torch.squeeze(self.max_pool(torch.cat((unigrams, bigrams, trigrams), 2)))
        phrase = phrase.permute(0,2,1)
        # 3rd level: sentence-level features
        self.encoder.flatten_parameters()
        sentence, _ = self.encoder(phrase) # out, (h,c)
        # final level: whole embedding
        concate = torch.cat((words, phrase, sentence), 2) # (batch_size, seq_len, module_dim * 2 + wordvec_dim)
        if self.self_attn == True:
            concate = self.self_attention(concate, concate, concate)
        question_embedding, output_embedding = self.concatRNN(concate, question_len, max_ques_len) # question_embedding: [Tensor] bs * max_seq_len * feat_dim, output_embedding: [Tensor] bs * feat_dim
        question_embedding = F.dropout(question_embedding, self.dropout, self.training) # batch_size * max_seq_len * feat_dim
        output_embedding = F.dropout(output_embedding, self.dropout, self.training) # batch_size * feat_dim

        return output_embedding, words, question_embedding

class VisualAppearanceEncoder(nn.Module): # 1 LSTM
    def __init__(self, appearance_dim = 2048, module_dim=512,bidirectional=True):
        super(VisualAppearanceEncoder, self).__init__()

        self.input_dim = appearance_dim  # 512


        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = module_dim // 2  # 256

        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(appearance_dim, rnn_dim, batch_first=False, bidirectional=bidirectional)  # 300， 256
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.finalvisual_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, appearance_clips):
        """
        Args:
            appearance_clips: [Tensor] (batch_size, num_of_clips, num_of_frames, feat_dim)
        return:
            appearance_clips representation [Tensor] (batch_size, num_of_clips, new_feat_dim)
        """
        bs = appearance_clips.size(0)
        feat_dim = appearance_clips.size(-1)
        num_of_frames = appearance_clips.size(2)

        embed = self.tanh(self.embedding_dropout(appearance_clips))  # tanh+dropout
        embed = torch.transpose(embed,0,2).contiguous()
        embed = torch.transpose(embed,1,2).contiguous() # (num_of_frames * batch_size * num_of_clips * feat_dim)
        embed = embed.view(num_of_frames,-1,feat_dim)


        self.encoder.flatten_parameters()  # 为了提高内存的利用率和效率, 我们调用使parameter的数据存放变成连续的块
        _, (visual_appearance_embedding, _) = self.encoder(embed)  # out, (h,c)
        if self.bidirectional:
            visual_appearance_embedding = torch.cat([visual_appearance_embedding[0], visual_appearance_embedding[1]], -1)  # 最后一个状态

        visual_appearance_embedding = self.finalvisual_dropout(visual_appearance_embedding)
        visual_appearance_embedding = visual_appearance_embedding.view(bs,-1,self.module_dim)

        return visual_appearance_embedding


