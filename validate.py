import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse
import os, sys
import json
import pickle
from termcolor import colored

from DataLoader import VideoQADataLoader
from utils import todevice

import model.models as modelset

from config import cfg, cfg_from_file

QUESTION_CATEGORY = {0:'count',1:'exist',2:'query_color',3:'query_size',4:'query_actiontype',5:'query_direction',
                     6:'query_shape',7:'compare_more',8:'compare_equal',9:'compare_less',10:'attribute_compare_color',
                     11:'attribute_compare_size',12:'attribute_compare_actiontype',13:'attribute_compare_direction',
                     14:'attribute_compare_shape'}

def validate(cfg, model, data, device, write_preds=False):
    model.eval()
    print('validating...')
    total_acc, count = 0.0, 0
    all_preds = []
    gts = []
    v_ids = []
    q_ids = []
    if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
        what_acc,who_acc,how_acc,when_acc,where_acc = 0.,0.,0.,0.,0.
        what_count, who_count, how_count, when_count, where_count = 0,0,0,0,0
    elif cfg.dataset.name == 'svqa':
        count_acc, exist_acc, query_color_acc, query_size_acc, query_actiontype_acc, \
        query_direction_acc, query_shape_acc, compare_more_acc, compare_equal_acc, compare_less_acc, \
        attribute_compare_color_acc, attribute_compare_size_acc, attribute_compare_actiontype_acc, \
        attribute_compare_direction_acc, attribute_compare_shape_acc = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        count_ct, exist_ct, query_color_ct, query_size_ct, query_actiontype_ct, query_direction_ct, \
        query_shape_ct, compare_more_ct, compare_equal_ct, compare_less_ct, attribute_compare_color_ct, \
        attribute_compare_size_ct, attribute_compare_actiontype_ct, attribute_compare_direction_ct, \
        attribute_compare_shape_ct = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            if cfg.dataset.name == 'svqa':
                video_ids, question_ids, question_categories, answers, *batch_input = [todevice(x, device) for x in batch]
            else:
                video_ids, question_ids, answers, *batch_input = [todevice(x, device) for x in batch]
            if cfg.train.batch_size == 1:
                answers = answers.to(device)
            else:
                answers = answers.to(device).squeeze()
            if cfg.model_type == 'DualVGR':
                logits, aq_embed, mq_embed, com_app, com_motion, aq_fusion, mq_fusion = model(*batch_input) #attn,appear_scores,mot_scores,
            else:
                logits = model(*batch_input)
            
            preds = logits.detach().argmax(1)
            agreeings = (preds == answers)
            if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
                what_idx = []
                who_idx = []
                how_idx = []
                when_idx = []
                where_idx = []

                key_word = batch_input[-2][:,0].to('cpu') # batch-based questions word
                for i,word in enumerate(key_word):
                    word = int(word)
                    if data.vocab['question_idx_to_token'][word] == 'what':
                        what_idx.append(i)
                    elif data.vocab['question_idx_to_token'][word] == 'who':
                        who_idx.append(i)
                    elif data.vocab['question_idx_to_token'][word] == 'how':
                        how_idx.append(i)
                    elif data.vocab['question_idx_to_token'][word] == 'when':
                        when_idx.append(i)
                    elif data.vocab['question_idx_to_token'][word] == 'where':
                        where_idx.append(i)
            else:
                count_idx = []
                exist_idx = []
                query_color_idx = []
                query_size_idx = []
                query_actiontype_idx = []
                query_direction_idx = []
                query_shape_idx = []
                compare_more_idx = []
                compare_equal_idx = []
                compare_less_idx = []
                attribute_compare_color_idx = []
                attribute_compare_size_idx = []
                attribute_compare_actiontype_idx = []
                attribute_compare_direction_idx = []
                attribute_compare_shape_idx = []
                for i, category in enumerate(question_categories):
                    category = int(category.cpu())
                    if QUESTION_CATEGORY[category] == 'count':
                        count_idx.append(i)
                    elif QUESTION_CATEGORY[category] == 'exist':
                        exist_idx.append(i)
                    elif QUESTION_CATEGORY[category] == 'query_color':
                        query_color_idx.append(i)
                    elif QUESTION_CATEGORY[category] == 'query_size':
                        query_size_idx.append(i)
                    elif QUESTION_CATEGORY[category] == 'query_actiontype':
                        query_actiontype_idx.append(i)
                    elif QUESTION_CATEGORY[category] == 'query_direction':
                        query_direction_idx.append(i)
                    elif QUESTION_CATEGORY[category] == 'query_shape':
                        query_shape_idx.append(i)
                    elif QUESTION_CATEGORY[category] == 'compare_more':
                        compare_more_idx.append(i)
                    elif QUESTION_CATEGORY[category] == 'compare_equal':
                        compare_equal_idx.append(i)
                    elif QUESTION_CATEGORY[category] == 'compare_less':
                        compare_less_idx.append(i)
                    elif QUESTION_CATEGORY[category] == 'attribute_compare_color':
                        attribute_compare_color_idx.append(i)
                    elif QUESTION_CATEGORY[category] == 'attribute_compare_size':
                        attribute_compare_size_idx.append(i)
                    elif QUESTION_CATEGORY[category] == 'attribute_compare_actiontype':
                        attribute_compare_actiontype_idx.append(i)
                    elif QUESTION_CATEGORY[category] == 'attribute_compare_direction':
                        attribute_compare_direction_idx.append(i)
                    elif QUESTION_CATEGORY[category] == 'attribute_compare_shape':
                        attribute_compare_shape_idx.append(i)
                    else:
                        raise ValueError('unseen value in question categories?')
         
            
            if write_preds:
                preds = logits.argmax(1)
                answer_vocab = data.vocab['answer_idx_to_token']

                for predict in preds:
                    all_preds.append(answer_vocab[predict.item()])
                
                for gt in answers:
                    gts.append(answer_vocab[gt.item()])
                
                for id in video_ids:
                    v_ids.append(id.cpu().numpy())
                for ques_id in question_ids:
                    q_ids.append(ques_id)

            if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
                total_acc += agreeings.float().sum().item()
                count += answers.size(0)

                what_acc += agreeings.float()[what_idx].sum().item() if what_idx != [] else 0
                who_acc += agreeings.float()[who_idx].sum().item() if who_idx != [] else 0
                how_acc += agreeings.float()[how_idx].sum().item() if how_idx != [] else 0
                when_acc += agreeings.float()[when_idx].sum().item() if when_idx != [] else 0
                where_acc += agreeings.float()[where_idx].sum().item() if where_idx != [] else 0
                what_count += len(what_idx)
                who_count += len(who_idx)
                how_count += len(how_idx)
                when_count += len(when_idx)
                where_count += len(where_idx)
            else:
                total_acc += agreeings.float().sum().item()
                count += answers.size(0)

                count_acc += agreeings.float()[count_idx].sum().item() if count_idx != [] else 0
                exist_acc += agreeings.float()[exist_idx].sum().item() if exist_idx != [] else 0
                query_color_acc += agreeings.float()[query_color_idx].sum().item() if query_color_idx != [] else 0
                query_size_acc += agreeings.float()[query_size_idx].sum().item() if query_size_idx != [] else 0
                query_actiontype_acc += agreeings.float()[query_actiontype_idx].sum().item() if query_actiontype_idx != [] else 0
                query_direction_acc += agreeings.float()[query_direction_idx].sum().item() if query_direction_idx != [] else 0
                query_shape_acc += agreeings.float()[query_shape_idx].sum().item() if query_shape_idx != [] else 0
                compare_more_acc += agreeings.float()[compare_more_idx].sum().item() if compare_more_idx != [] else 0
                compare_equal_acc += agreeings.float()[compare_equal_idx].sum().item() if compare_equal_idx != [] else 0
                compare_less_acc += agreeings.float()[compare_less_idx].sum().item() if compare_less_idx != [] else 0
                attribute_compare_color_acc += agreeings.float()[attribute_compare_color_idx].sum().item() if attribute_compare_color_idx != [] else 0
                attribute_compare_size_acc += agreeings.float()[
                    attribute_compare_size_idx].sum().item() if attribute_compare_size_idx != [] else 0
                attribute_compare_actiontype_acc += agreeings.float()[
                    attribute_compare_actiontype_idx].sum().item() if attribute_compare_actiontype_idx != [] else 0
                attribute_compare_direction_acc += agreeings.float()[
                    attribute_compare_direction_idx].sum().item() if attribute_compare_direction_idx != [] else 0
                attribute_compare_shape_acc += agreeings.float()[
                    attribute_compare_shape_idx].sum().item() if attribute_compare_shape_idx != [] else 0

                count_ct += len(count_idx)
                exist_ct += len(exist_idx)
                query_color_ct += len(query_color_idx)
                query_size_ct += len(query_size_idx)
                query_actiontype_ct += len(query_actiontype_idx)
                query_direction_ct += len(query_direction_idx)
                query_shape_ct += len(query_shape_idx)
                compare_more_ct += len(compare_more_idx)
                compare_equal_ct += len(compare_equal_idx)
                compare_less_ct += len(compare_less_idx)
                attribute_compare_color_ct += len(attribute_compare_color_idx)
                attribute_compare_size_ct += len(attribute_compare_size_idx)
                attribute_compare_actiontype_ct += len(attribute_compare_actiontype_idx)
                attribute_compare_direction_ct += len(attribute_compare_direction_idx)
                attribute_compare_shape_ct += len(attribute_compare_shape_idx)

        acc = total_acc / count
        if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
            what_acc = what_acc / what_count
            who_acc = who_acc / who_count
            how_acc = how_acc / how_count
            when_acc = when_acc / when_count
            where_acc = where_acc / where_count
        else:
            count_acc = count_acc / count_ct
            exist_acc = exist_acc / exist_ct
            query_color_acc = query_color_acc / query_color_ct
            query_size_acc = query_size_acc / query_size_ct
            query_actiontype_acc = query_actiontype_acc / query_actiontype_ct
            query_direction_acc = query_direction_acc / query_direction_ct
            query_shape_acc = query_shape_acc / query_shape_ct
            compare_more_acc = compare_more_acc / compare_more_ct
            compare_equal_acc = compare_equal_acc / compare_equal_ct
            compare_less_acc = compare_less_acc / compare_less_ct
            attribute_compare_color_acc = attribute_compare_color_acc / attribute_compare_color_ct
            attribute_compare_size_acc = attribute_compare_size_acc / attribute_compare_size_ct
            attribute_compare_actiontype_acc = attribute_compare_actiontype_acc / attribute_compare_actiontype_ct
            attribute_compare_direction_acc = attribute_compare_direction_acc / attribute_compare_direction_ct
            attribute_compare_shape_acc = attribute_compare_shape_acc / attribute_compare_shape_ct
   
    if not write_preds:
        if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
            return acc, what_acc, who_acc, how_acc, when_acc, where_acc
        else:
            return acc, count_acc, exist_acc, query_color_acc, query_size_acc, query_actiontype_acc, query_direction_acc, query_shape_acc, compare_more_acc, compare_equal_acc, compare_less_acc, attribute_compare_color_acc, attribute_compare_size_acc, attribute_compare_actiontype_acc, attribute_compare_direction_acc, attribute_compare_shape_acc
    else:
        if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
            return acc, all_preds, gts, v_ids, q_ids, what_acc, who_acc, how_acc, when_acc, where_acc
        else:
            return acc, all_preds, gts, v_ids, q_ids, count_acc, exist_acc, query_color_acc, query_size_acc, query_actiontype_acc, query_direction_acc, query_shape_acc, compare_more_acc, compare_equal_acc, compare_less_acc, attribute_compare_color_acc, attribute_compare_size_acc, attribute_compare_actiontype_acc, attribute_compare_direction_acc, attribute_compare_shape_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='msvdqa_DualVGR.yml', type=str)
    parser.add_argument('--unit_layers', dest='unit_layers',help='unit_layers', default=1,type=int)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    assert cfg.dataset.name in ['svqa', 'msrvtt-qa', 'msvd-qa']
    # check if the data folder exists
    assert os.path.exists(cfg.dataset.data_dir)

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt','common_specific_gatepunishfinal12021-01-24_Sunday_10:08:22_model.pt') # TODO
    assert os.path.exists(ckpt)
    # load pretrained model
    loaded = torch.load(ckpt, map_location='cpu')
    model_kwargs = loaded['model_kwargs']

    cfg.dataset.appearance_feat = '{}_appearance_feat.h5'
    cfg.dataset.motion_feat = '{}_motion_feat.h5'
    cfg.dataset.vocab_json = '{}_vocab.json'
    cfg.dataset.test_question_pt = '{}_test_questions.pt'

    cfg.dataset.test_question_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_pt.format(cfg.dataset.name))
    cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name))

    cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name))
    cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name))

    test_loader_kwargs = {
        'question_pt': cfg.dataset.test_question_pt,
        'vocab_json': cfg.dataset.vocab_json,
        'appearance_feat': cfg.dataset.appearance_feat,
        'motion_feat': cfg.dataset.motion_feat,
        'test_num': cfg.test.test_num,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'shuffle': False
    }
    test_loader = VideoQADataLoader(**test_loader_kwargs)
    model_kwargs.update({'vocab': test_loader.vocab})
    model_kwargs['unit_layers'] = args.unit_layers
    if cfg.model_type == 'DualVGR':
        model = modelset.DualVGR(**model_kwargs).to(device)
    
    model.load_state_dict(loaded['state_dict'])

    if cfg.test.write_preds:
        acc, preds, gts, v_ids, q_ids, *test_output = validate(cfg, model, test_loader, device, cfg.test.write_preds)
        if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
            sys.stdout.write(
                '~~~~~~ Test Accuracy: {valid_acc}, What Accuracy: {what_acc}, Who Accuracy: {who_acc}, How Accuracy: {how_acc}, When Accuracy: {when_acc}, Where Accuracy: {where_acc} ~~~~~~~\n'.format(
                    valid_acc=colored("{:.4f}".format(acc), "red", attrs=['bold']),
                    what_acc=colored("{:.4f}".format(test_output[0]), "red", attrs=['bold']),
                    who_acc=colored('{:.4f}'.format(test_output[1]), "red", attrs=['bold']),
                    how_acc=colored('{:.4f}'.format(test_output[2]), "red", attrs=['bold']),
                    when_acc=colored('{:.4f}'.format(test_output[3]), "red", attrs=['bold']),
                    where_acc=colored('{:.4f}'.format(test_output[4]), "red", attrs=['bold'])
                ))
            sys.stdout.flush()
        else:
            sys.stdout.write(
                '~~~~~~ Test Accuracy: {valid_acc}, Count Accuracy: {count_acc}, Exist Accuracy: {exist_acc}, Query_Color Accuracy: {query_color_acc}, '
                'Query_Size Accuracy: {query_size_acc}, Query_Actiontype Accuracy: {query_actiontype_acc}, Query_Direction Accuracy: {query_direction_acc},'
                'Query_Shape Accuracy: {query_shape_acc}, Compare_More Accuracy: {compare_more_acc}, Compare_Equal Accuracy: {compare_equal_acc}, '
                'Compare_Less Accuracy: {compare_less_acc}, Attribute_Compare_Color Accuracy: {attribute_compare_color_acc}, Attribute_Compare_Size Accuracy: {attribute_compare_size_acc},'
                'Attribute_Compare_Actiontype Accuracy: {attribute_compare_actiontype_acc}, Attribute_Compare_Direction Accuracy: {attribute_compare_direction_acc},'
                'Attribute_Compare_Shape Accuracy: {attribute_compare_shape_acc} ~~~~~~~\n'.format(
                    valid_acc=colored("{:.4f}".format(acc), "red", attrs=['bold']),
                    count_acc=colored("{:.4f}".format(test_output[0]), "red", attrs=['bold']),
                    exist_acc=colored('{:.4f}'.format(test_output[1]), "red", attrs=['bold']),
                    query_color_acc=colored('{:.4f}'.format(test_output[2]), "red", attrs=['bold']),
                    query_size_acc=colored('{:.4f}'.format(test_output[3]), "red", attrs=['bold']),
                    query_actiontype_acc=colored('{:.4f}'.format(test_output[4]), "red", attrs=['bold']),
                    query_direction_acc=colored('{:.4f}'.format(test_output[5]), "red", attrs=['bold']),
                    query_shape_acc=colored('{:.4f}'.format(test_output[6]), "red", attrs=['bold']),
                    compare_more_acc=colored('{:.4f}'.format(test_output[7]), "red", attrs=['bold']),
                    compare_equal_acc=colored('{:.4f}'.format(test_output[8]), "red", attrs=['bold']),
                    compare_less_acc=colored('{:.4f}'.format(test_output[9]), "red", attrs=['bold']),
                    attribute_compare_color_acc=colored('{:.4f}'.format(test_output[10]), "red", attrs=['bold']),
                    attribute_compare_size_acc=colored('{:.4f}'.format(test_output[11]), "red", attrs=['bold']),
                    attribute_compare_actiontype_acc=colored('{:.4f}'.format(test_output[12]), "red", attrs=['bold']),
                    attribute_compare_direction_acc=colored('{:.4f}'.format(test_output[13]), "red", attrs=['bold']),
                    attribute_compare_shape_acc=colored('{:.4f}'.format(test_output[14]), "red", attrs=['bold'])
                ))
            sys.stdout.flush()

        # write predictions for visualization purposes
        output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            assert os.path.isdir(output_dir)
        preds_file = os.path.join(output_dir, "test_preds.json")

        
        vocab = test_loader.vocab['question_idx_to_token']
        dict = {}
        with open(cfg.dataset.test_question_pt, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            org_v_ids = obj['video_ids']
            org_v_names = obj['video_names']
            org_q_ids = obj['question_id']

        for idx in range(len(org_q_ids)):
            dict[str(org_q_ids[idx])] = [org_v_names[idx], questions[idx]]
        instances = [
            {'video_id': video_id, 'question_id': q_id, 'video_name': str(dict[str(q_id)][0]), 'question': [vocab[word.item()] for word in dict[str(q_id)][1] if word != 0],
             'answer': answer,
             'prediction': pred} for video_id, q_id, answer, pred in
            zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds)]
        # write preditions to json file
        with open(preds_file, 'w') as f:
            json.dump(instances, f)
        sys.stdout.write('Display 10 samples...\n')
        # Display 10 examples
        for idx in range(10):
            print('Video name: {}'.format(dict[str(q_ids[idx].item())][0]))
            cur_question = [vocab[word.item()] for word in dict[str(q_ids[idx].item())][1] if word != 0]
            print('Question: ' + ' '.join(cur_question) + '?')
            print('Prediction: {}'.format(preds[idx]))
            print('Groundtruth: {}'.format(gts[idx]))
    else:
        acc, *test_output = validate(cfg, model, test_loader, device, cfg.test.write_preds)
        if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
            sys.stdout.write(
                '~~~~~~ Test Accuracy: {valid_acc}, What Accuracy: {what_acc}, Who Accuracy: {who_acc}, How Accuracy: {how_acc}, When Accuracy: {when_acc}, Where Accuracy: {where_acc} ~~~~~~~\n'.format(
                    valid_acc=colored("{:.4f}".format(acc), "red", attrs=['bold']),
                    what_acc=colored("{:.4f}".format(test_output[0]), "red", attrs=['bold']),
                    who_acc=colored('{:.4f}'.format(test_output[1]), "red", attrs=['bold']),
                    how_acc=colored('{:.4f}'.format(test_output[2]), "red", attrs=['bold']),
                    when_acc=colored('{:.4f}'.format(test_output[3]), "red", attrs=['bold']),
                    where_acc=colored('{:.4f}'.format(test_output[4]), "red", attrs=['bold'])
                ))
            sys.stdout.flush()
        else:
            sys.stdout.write(
                '~~~~~~ Test Accuracy: {valid_acc}, Count Accuracy: {count_acc}, Exist Accuracy: {exist_acc}, Query_Color Accuracy: {query_color_acc}, '
                'Query_Size Accuracy: {query_size_acc}, Query_Actiontype Accuracy: {query_actiontype_acc}, Query_Direction Accuracy: {query_direction_acc},'
                'Query_Shape Accuracy: {query_shape_acc}, Compare_More Accuracy: {compare_more_acc}, Compare_Equal Accuracy: {compare_equal_acc}, '
                'Compare_Less Accuracy: {compare_less_acc}, Attribute_Compare_Color Accuracy: {attribute_compare_color_acc}, Attribute_Compare_Size Accuracy: {attribute_compare_size_acc},'
                'Attribute_Compare_Actiontype Accuracy: {attribute_compare_actiontype_acc}, Attribute_Compare_Direction Accuracy: {attribute_compare_direction_acc},'
                'Attribute_Compare_Shape Accuracy: {attribute_compare_shape_acc} ~~~~~~~\n'.format(
                    valid_acc=colored("{:.4f}".format(acc), "red", attrs=['bold']),
                    count_acc=colored("{:.4f}".format(test_output[0]), "red", attrs=['bold']),
                    exist_acc=colored('{:.4f}'.format(test_output[1]), "red", attrs=['bold']),
                    query_color_acc=colored('{:.4f}'.format(test_output[2]), "red", attrs=['bold']),
                    query_size_acc=colored('{:.4f}'.format(test_output[3]), "red", attrs=['bold']),
                    query_actiontype_acc=colored('{:.4f}'.format(test_output[4]), "red", attrs=['bold']),
                    query_direction_acc=colored('{:.4f}'.format(test_output[5]), "red", attrs=['bold']),
                    query_shape_acc=colored('{:.4f}'.format(test_output[6]), "red", attrs=['bold']),
                    compare_more_acc=colored('{:.4f}'.format(test_output[7]), "red", attrs=['bold']),
                    compare_equal_acc=colored('{:.4f}'.format(test_output[8]), "red", attrs=['bold']),
                    compare_less_acc=colored('{:.4f}'.format(test_output[9]), "red", attrs=['bold']),
                    attribute_compare_color_acc=colored('{:.4f}'.format(test_output[10]), "red", attrs=['bold']),
                    attribute_compare_size_acc=colored('{:.4f}'.format(test_output[11]), "red", attrs=['bold']),
                    attribute_compare_actiontype_acc=colored('{:.4f}'.format(test_output[12]), "red", attrs=['bold']),
                    attribute_compare_direction_acc=colored('{:.4f}'.format(test_output[13]), "red", attrs=['bold']),
                    attribute_compare_shape_acc=colored('{:.4f}'.format(test_output[14]), "red", attrs=['bold'])
                ))
            sys.stdout.flush()
