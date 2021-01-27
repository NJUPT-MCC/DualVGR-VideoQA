# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import numpy as np
import json
import pickle
import torch
import math
import h5py
from torch.utils.data import Dataset, DataLoader

QUESTION_CATEGORY = {'count':0, 'exist':1, 'query_color':2, 'query_size':3, 'query_actiontype':4, 'query_actiondir':5, 'query_shape':6, 'greater_than':7, 'equal_to':8,
        'less_than':9, 'equal_color':10, 'equal_size':11, 'equal_actiontype':12, 'equal_actiondir':13, 'equal_shape':14}

def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    return vocab


class VideoQADataset(Dataset):

    def __init__(self, answers, questions, questions_len, video_ids, q_ids,
                 app_feature_h5, app_feat_id_to_index, motion_feature_h5, motion_feat_id_to_index, question_category=None):
        # convert data to tensor
        self.all_answers = answers
        self.question_category = question_category
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))
        self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
        self.all_q_ids = q_ids
        self.app_feature_h5 = app_feature_h5
        self.motion_feature_h5 = motion_feature_h5
        self.app_feat_id_to_index = app_feat_id_to_index
        self.motion_feat_id_to_index = motion_feat_id_to_index

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None

        question = self.all_questions[index]
        question_category = QUESTION_CATEGORY[self.question_category[index]] if self.question_category is not None else None
        question_len = self.all_questions_len[index]
        video_idx = self.all_video_ids[index].item()
        question_idx = self.all_q_ids[index]
        app_index = self.app_feat_id_to_index[str(video_idx)]
        motion_index = self.motion_feat_id_to_index[str(video_idx)]
        with h5py.File(self.app_feature_h5, 'r') as f_app:
            appearance_feat = f_app['resnet_features'][app_index]  # (8, 16, 2048)
        with h5py.File(self.motion_feature_h5, 'r') as f_motion:
            motion_feat = f_motion['resnext_features'][motion_index]  # (8, 2048)
        appearance_feat = torch.from_numpy(appearance_feat)
        motion_feat = torch.from_numpy(motion_feat)
        if question_category == None:
            return (
                    video_idx, question_idx, answer, appearance_feat, motion_feat, question,
                    question_len)
        else:
            return (
            video_idx, question_idx, question_category, answer, appearance_feat, motion_feat, question,
            question_len)

    def __len__(self):
        return len(self.all_questions)


class VideoQADataLoader(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path) # id-token
        question_category = None
        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            if 'question_category' in obj:
                question_category = obj['question_category']
            questions = obj['questions']
            questions_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers'] # vocab-ans id
            glove_matrix = obj['glove']

        if 'train_num' in kwargs: # select subset
            trained_num = kwargs.pop('train_num')
            if trained_num > 0:
                questions = questions[:trained_num]
                question_category = question_category[:trained_num]
                questions_len = questions_len[:trained_num]
                video_ids = video_ids[:trained_num]
                q_ids = q_ids[:trained_num]
                answers = answers[:trained_num]

        if 'val_num' in kwargs: # select subset
            val_num = kwargs.pop('val_num')
            if val_num > 0:
                questions = questions[:val_num]
                question_category = question_category[:val_num]
                questions_len = questions_len[:val_num]
                video_ids = video_ids[:val_num]
                q_ids = q_ids[:val_num]
                answers = answers[:val_num]

        if 'test_num' in kwargs: # select subset
            test_num = kwargs.pop('test_num')
            if test_num > 0:
                questions = questions[:test_num]
                question_category = question_category[:test_num]
                questions_len = questions_len[:test_num]
                video_ids = video_ids[:test_num]
                q_ids = q_ids[:test_num]
                answers = answers[:test_num]

        print('loading appearance feature from %s' % (kwargs['appearance_feat']))
        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]
        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}
        print('loading motion feature from %s' % (kwargs['motion_feat']))
        with h5py.File(kwargs['motion_feat'], 'r') as motion_features_file:
            motion_video_ids = motion_features_file['ids'][()]
        motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}
        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.motion_feature_h5 = kwargs.pop('motion_feat')
        if question_category == None:
            self.dataset = VideoQADataset(answers, questions, questions_len,
                                    video_ids, q_ids,
                                    self.app_feature_h5, app_feat_id_to_index, self.motion_feature_h5,
                                    motion_feat_id_to_index)
        else:
            self.dataset = VideoQADataset(answers, questions, questions_len,
                                          video_ids, q_ids,
                                          self.app_feature_h5, app_feat_id_to_index, self.motion_feature_h5,
                                          motion_feat_id_to_index, question_category)

        self.vocab = vocab
        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix # 对应于question-tokenid

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
