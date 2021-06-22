import os, sys

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import time
import logging
from termcolor import colored

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from DataLoader import VideoQADataLoader
from utils import *
from validate import validate

import model.models as modelset


from config import cfg, cfg_from_file

lctime = time.localtime()
lctime = time.strftime("%Y-%m-%d_%A_%H:%M:%S",lctime)
def train(cfg):
    logging.info("Create train_loader and val_loader.........")
    train_loader_kwargs = {
            'question_pt': cfg.dataset.train_question_pt,
            'vocab_json': cfg.dataset.vocab_json,
            'appearance_feat': cfg.dataset.appearance_feat,
            'motion_feat': cfg.dataset.motion_feat,
            'train_num': cfg.train.train_num,
            'batch_size': cfg.train.batch_size,
            'num_workers': cfg.num_workers,
            'shuffle': True
        }
    train_loader = VideoQADataLoader(**train_loader_kwargs)
    logging.info("number of train instances: {}".format(len(train_loader.dataset)))
    if cfg.val.flag:
        val_loader_kwargs = {
            'question_pt': cfg.dataset.val_question_pt,
            'vocab_json': cfg.dataset.vocab_json,
            'appearance_feat': cfg.dataset.appearance_feat, # h5
            'motion_feat': cfg.dataset.motion_feat,
            'val_num': cfg.val.val_num,
            'batch_size': cfg.train.batch_size,
            'num_workers': cfg.num_workers,
            'shuffle': False
        }
        val_loader = VideoQADataLoader(**val_loader_kwargs)
        logging.info("number of val instances: {}".format(len(val_loader.dataset)))

    # Create the model
    logging.info("Create model.........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {
        'vision_dim': cfg.train.vision_dim,
        'module_dim': cfg.train.module_dim,
        'word_dim': cfg.train.word_dim,
        'vocab': train_loader.vocab,
        'num_of_nodes': cfg.train.num_of_nodes,
        'graph_module': cfg.graph_module,
        'graph_layers': cfg.graph_layers
    }
    model_kwargs_tosave = {k: v for k, v in model_kwargs.items() if k != 'vocab'}
    if cfg.model_type == 'DualVGR':
        model = modelset.DualVGR(**model_kwargs).to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('num of params: {}'.format(pytorch_total_params))
    logging.info(model)

    if cfg.train.glove: # set glove
        logging.info('load glove vectors')
        train_loader.glove_matrix = torch.FloatTensor(train_loader.glove_matrix).to(device)
        with torch.no_grad():
            model.linguistic_input_unit.encoder_embed.weight.set_(train_loader.glove_matrix)
    if torch.cuda.device_count() > 1 and cfg.multi_gpus:
        model = model.cuda()
        logging.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=None)

    optimizer = optim.Adam(model.parameters(), cfg.train.lr)

    start_epoch = 0
    if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
        best_val = 0.
        best_what = 0.
        best_how = 0.
        best_when = 0.
        best_who = 0.
        best_where = 0.
    else:
        best_val = 0.
        best_count = 0.
        best_exist = 0.
        best_query_color = 0.
        best_query_size = 0.
        best_query_actiontype = 0.
        best_query_direction = 0.
        best_query_shape = 0.
        best_compare_more = 0.
        best_compare_equal = 0.
        best_compare_less = 0.
        best_attribute_compare_color = 0.
        best_attribute_compare_size = 0.
        best_attribute_compare_actiontype = 0.
        best_attribute_compare_direction = 0.
        best_attribute_compare_shape = 0.

    if cfg.train.restore:
        print("Restore checkpoint and optimizer...")
        ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model.pt') # TODO
        ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    criterion = nn.CrossEntropyLoss().to(device)

    logging.info("Start training........")
    for epoch in range(start_epoch, cfg.train.max_epochs):
        logging.info('>>>>>> epoch {epoch} <<<<<<'.format(epoch=colored("{}".format(epoch), "green", attrs=["bold"])))
        model.train()
        total_acc, count = 0, 0
        total_loss, avg_loss = 0.0, 0.0
        avg_loss = 0
        train_accuracy = 0
        for i, batch in enumerate(iter(train_loader)):
            progress = epoch + i / len(train_loader)
            if cfg.dataset.name == 'svqa':
                _, _, question_categories, answers, *batch_input = [todevice(x, device) for x in batch]
            else:
                _, _, answers, *batch_input = [todevice(x, device) for x in batch]

            answers = answers.cuda().squeeze()
            optimizer.zero_grad()
            if cfg.model_type == 'DualVGR':
                logits, aq_embed, mq_embed, com_app, com_motion, aq_fusion, mq_fusion = model(
                    *batch_input)  # batch_input-batchsize*attn,appear_scores,mot_scores,
            else:
                logits = model(*batch_input)
            
            loss = criterion(logits, answers)
            if cfg.model_type == 'DualVGR':
                loss_dep = 0
                loss_com = 0
                temp = len(aq_fusion)
                for i in range(temp):
                    loss_dep += (loss_dependence(aq_fusion[i].cuda(),com_app[i].cuda(),cfg.train.num_of_nodes)+loss_dependence(mq_fusion[i].cuda(),com_motion[i].cuda(),cfg.train.num_of_nodes))
                    loss_com += common_loss(com_app[i].cuda(),com_motion[i].cuda())
                loss = loss + cfg.alpha * loss_com/temp + cfg.beta * loss_dep/temp
            loss.backward()
            total_loss += loss.detach()
            avg_loss = total_loss / (i + 1)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
            optimizer.step()
            aggreeings = batch_accuracy(logits, answers)

        # Training Phase

            total_acc += aggreeings.sum().item() # 正确的个数
            count += answers.size(0) # 答案
            train_accuracy = total_acc / count
            sys.stdout.write(
                "\rProgress = {progress}   ce_loss = {ce_loss}   avg_loss = {avg_loss}    train_acc = {train_acc}    avg_acc = {avg_acc}    exp: {exp_name}".format(
                    progress=colored("{:.3f}".format(progress), "green", attrs=['bold']),
                    ce_loss=colored("{:.4f}".format(loss.item()), "blue", attrs=['bold']),
                    avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold']),
                    train_acc=colored("{:.4f}".format(aggreeings.float().mean().cpu().numpy()), "blue",
                                        attrs=['bold']),
                    avg_acc=colored("{:.4f}".format(train_accuracy), "red", attrs=['bold']),
                    exp_name=cfg.exp_name))
            sys.stdout.flush()

        sys.stdout.write("\n")
        if (epoch + 1) % 10 == 0:
            optimizer = step_decay(cfg, optimizer)
        sys.stdout.flush()
        logging.info("Epoch = %s   avg_loss = %.3f    avg_acc = %.3f" % (epoch, avg_loss, train_accuracy))

        if cfg.val.flag:
            output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
                assert os.path.isdir(output_dir)

            if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
                valid_acc, *valid_output = validate(cfg, model, val_loader, device, write_preds=False)
                if (valid_acc > best_val):
                    best_val = valid_acc
                    best_what = valid_output[0]
                    best_who = valid_output[1]
                    best_when = valid_output[3]
                    best_how = valid_output[2]
                    best_where = valid_output[4]
                    # Save best model
                    ckpt_dir = os.path.join(cfg.dataset.save_dir, 'ckpt')
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    else:
                        assert os.path.isdir(ckpt_dir)

                    save_checkpoint(epoch, model, optimizer, model_kwargs_tosave, os.path.join(ckpt_dir, cfg.model_type+str(cfg.graph_layers)+lctime+'_model.pt'))
                    sys.stdout.write('\n >>>>>> save to %s <<<<<< \n' % (ckpt_dir))
                    sys.stdout.flush()

                logging.info('~~~~~~ Valid Accuracy: %.4f ~~~~~~~' % valid_acc)
                logging.info('~~~~~~ Valid What Accuracy: %.4f ~~~~~~~' % valid_output[0])
                logging.info('~~~~~~ Valid Who Accuracy: %.4f ~~~~~~' % valid_output[1])
                logging.info('~~~~~~ Valid How Accuracy: %.4f ~~~~~~' % valid_output[2])
                logging.info('~~~~~~ Valid When Accuracy: %.4f ~~~~~~' % valid_output[3])
                logging.info('~~~~~~ Valid Where Accuracy: %.4f ~~~~~~' % valid_output[4])

                sys.stdout.write('~~~~~~ Valid Accuracy: {valid_acc}, What Accuracy: {what_acc}, Who Accuracy: {who_acc}, How Accuracy: {how_acc}, When Accuracy: {when_acc}, Where Accuracy: {where_acc} ~~~~~~~\n'.format(
                    valid_acc=colored("{:.4f}".format(valid_acc), "red", attrs=['bold']),
                    what_acc=colored("{:.4f}".format(valid_output[0]), "red", attrs=['bold']),
                    who_acc=colored('{:.4f}'.format(valid_output[1]), "red", attrs=['bold']),
                    how_acc=colored('{:.4f}'.format(valid_output[2]), "red", attrs=['bold']),
                    when_acc=colored('{:.4f}'.format(valid_output[3]), "red", attrs=['bold']),
                    where_acc=colored('{:.4f}'.format(valid_output[4]), "red", attrs=['bold'])
                    ))
                sys.stdout.flush()
            elif cfg.dataset.name == 'svqa':
                valid_acc, *valid_output = validate(cfg, model, val_loader, device, write_preds=False)
                if (valid_acc > best_val):
                    best_val = valid_acc
                    best_count = valid_output[0]
                    best_exist = valid_output[1]
                    best_query_color = valid_output[2]
                    best_query_size = valid_output[3]
                    best_query_actiontype = valid_output[4]
                    best_query_direction = valid_output[5]
                    best_query_shape = valid_output[6]
                    best_compare_more = valid_output[7]
                    best_compare_equal = valid_output[8]
                    best_compare_less = valid_output[9]
                    best_attribute_compare_color = valid_output[10]
                    best_attribute_compare_size = valid_output[11]
                    best_attribute_compare_actiontype = valid_output[12]
                    best_attribute_compare_direction = valid_output[13]
                    best_attribute_compare_shape = valid_output[14]
                    # Save best model
                    ckpt_dir = os.path.join(cfg.dataset.save_dir, 'ckpt')
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    else:
                        assert os.path.isdir(ckpt_dir)

                    save_checkpoint(epoch, model, optimizer, model_kwargs_tosave,
                                    os.path.join(ckpt_dir, cfg.model_type + str(cfg.graph_layers)+lctime + '_model.pt'))
                    sys.stdout.write('\n >>>>>> save to %s <<<<<< \n' % (ckpt_dir))
                    sys.stdout.flush()

                logging.info('~~~~~~ Valid Accuracy: %.4f ~~~~~~~' % valid_acc)
                logging.info('~~~~~~ Valid Count Accuracy: %.4f ~~~~~~~' % valid_output[0])
                logging.info('~~~~~~ Valid Exist Accuracy: %.4f ~~~~~~' % valid_output[1])
                logging.info('~~~~~~ Valid Query Color Accuracy: %.4f ~~~~~~' % valid_output[2])
                logging.info('~~~~~~ Valid Query Size Accuracy: %.4f ~~~~~~' % valid_output[3])
                logging.info('~~~~~~ Valid Query Actiontype Accuracy: %.4f ~~~~~~' % valid_output[4])
                logging.info('~~~~~~ Valid Query Direction Accuracy: %.4f ~~~~~~' % valid_output[5])
                logging.info('~~~~~~ Valid Query Shape Accuracy: %.4f ~~~~~~' % valid_output[6])
                logging.info('~~~~~~ Valid Compare More Accuracy: %.4f ~~~~~~' % valid_output[7])
                logging.info('~~~~~~ Valid Compare Equal Accuracy: %.4f ~~~~~~' % valid_output[8])
                logging.info('~~~~~~ Valid Compare Less Accuracy: %.4f ~~~~~~' % valid_output[9])
                logging.info('~~~~~~ Valid Attribute Compare Color Accuracy: %.4f ~~~~~~' % valid_output[10])
                logging.info('~~~~~~ Valid Attribute Compare Size Accuracy: %.4f ~~~~~~' % valid_output[11])
                logging.info('~~~~~~ Valid Attribute Compare Actiontype Accuracy: %.4f ~~~~~~' % valid_output[12])
                logging.info('~~~~~~ Valid Attribute Compare Direction Accuracy: %.4f ~~~~~~' % valid_output[13])
                logging.info('~~~~~~ Valid Attribute Compare Shape Accuracy: %.4f ~~~~~~' % valid_output[14])



                sys.stdout.write(
                    '~~~~~~ Valid Accuracy: {valid_acc}, Count Accuracy: {count_acc}, Exist Accuracy: {exist_acc}, Query_Color Accuracy: {query_color_acc}, '
                    'Query_Size Accuracy: {query_size_acc}, Query_Actiontype Accuracy: {query_actiontype_acc}, Query_Direction Accuracy: {query_direction_acc},'
                    'Query_Shape Accuracy: {query_shape_acc}, Compare_More Accuracy: {compare_more_acc}, Compare_Equal Accuracy: {compare_equal_acc}, '
                    'Compare_Less Accuracy: {compare_less_acc}, Attribute_Compare_Color Accuracy: {attribute_compare_color_acc}, Attribute_Compare_Size Accuracy: {attribute_compare_size_acc},'
                    'Attribute_Compare_Actiontype Accuracy: {attribute_compare_actiontype_acc}, Attribute_Compare_Direction Accuracy: {attribute_compare_direction_acc},'
                    'Attribute_Compare_Shape Accuracy: {attribute_compare_shape_acc} ~~~~~~~\n'.format(
                        valid_acc=colored("{:.4f}".format(valid_acc), "red", attrs=['bold']),
                        count_acc=colored("{:.4f}".format(valid_output[0]), "red", attrs=['bold']),
                        exist_acc=colored('{:.4f}'.format(valid_output[1]), "red", attrs=['bold']),
                        query_color_acc=colored('{:.4f}'.format(valid_output[2]), "red", attrs=['bold']),
                        query_size_acc=colored('{:.4f}'.format(valid_output[3]), "red", attrs=['bold']),
                        query_actiontype_acc=colored('{:.4f}'.format(valid_output[4]), "red", attrs=['bold']),
                        query_direction_acc=colored('{:.4f}'.format(valid_output[5]), "red", attrs=['bold']),
                        query_shape_acc=colored('{:.4f}'.format(valid_output[6]), "red", attrs=['bold']),
                        compare_more_acc=colored('{:.4f}'.format(valid_output[7]), "red", attrs=['bold']),
                        compare_equal_acc=colored('{:.4f}'.format(valid_output[8]), "red", attrs=['bold']),
                        compare_less_acc=colored('{:.4f}'.format(valid_output[9]), "red", attrs=['bold']),
                        attribute_compare_color_acc=colored('{:.4f}'.format(valid_output[10]), "red", attrs=['bold']),
                        attribute_compare_size_acc=colored('{:.4f}'.format(valid_output[11]), "red", attrs=['bold']),
                        attribute_compare_actiontype_acc=colored('{:.4f}'.format(valid_output[12]), "red", attrs=['bold']),
                        attribute_compare_direction_acc=colored('{:.4f}'.format(valid_output[13]), "red", attrs=['bold']),
                        attribute_compare_shape_acc=colored('{:.4f}'.format(valid_output[14]), "red", attrs=['bold'])
                    ))
                sys.stdout.flush()

        if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
            logging.info('~~~~~ Best Valid Accuracy: %.4f ~~~~~' % best_val)
            logging.info('~~~~~ Best What Accuracy: %.4f ~~~~~' % best_what)
            logging.info('~~~~~ Best How Accuracy: %.4f ~~~~~' % best_how)
            logging.info('~~~~~ Best When Accuracy: %.4f ~~~~~' % best_when)
            logging.info('~~~~~ Best Where Accuracy: %.4f ~~~~~'% best_where)
            logging.info('~~~~~ Best Who Accuracy: %.4f ~~~~~~' % best_who)
        else:
            logging.info('~~~~~ Best Valid Accuracy: %.4f ~~~~~' % best_val)
            logging.info('~~~~~ Best Count Accuracy: %.4f ~~~~~' % best_count)
            logging.info('~~~~~ Best Exist Accuracy: %.4f ~~~~~' % best_exist)
            logging.info('~~~~~ Best Query_Color Accuracy: %.4f ~~~~~' % best_query_color)
            logging.info('~~~~~ Best Query_Size Accuracy: %.4f ~~~~~' % best_query_size)
            logging.info('~~~~~ Best Query_Actiontype Accuracy: %.4f ~~~~~' % best_query_actiontype)
            logging.info('~~~~~ Best Query_Direction Accuracy: %.4f ~~~~~' % best_query_direction)
            logging.info('~~~~~ Best Query_Shape Accuracy: %.4f ~~~~~' % best_query_shape)
            logging.info('~~~~~ Best Compare_More Accuracy: %.4f ~~~~~' % best_compare_more)
            logging.info('~~~~~ Best Compare_Equal Accuracy: %.4f ~~~~~' % best_compare_equal)
            logging.info('~~~~~ Best Compare_Less Accuracy: %.4f ~~~~~' % best_compare_less)
            logging.info('~~~~~ Best Attribute_Compare_Color Accuracy: %.4f ~~~~~' % best_attribute_compare_color)
            logging.info('~~~~~ Best Attribute_Compare_Size Accuracy: %.4f ~~~~~' % best_attribute_compare_size)
            logging.info('~~~~~ Best Attribute_Compare_Actiontype Accuracy: %.4f ~~~~~' % best_attribute_compare_actiontype)
            logging.info('~~~~~ Best Attribute_Compare_Direction Accuracy: %.4f ~~~~~' % best_attribute_compare_direction)
            logging.info('~~~~~ Best Attribute_Compare_Shape Accuracy: %.4f ~~~~~' % best_attribute_compare_shape)




# Credit https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)


def step_decay(cfg, optimizer):
    # compute the new learning rate based on decay rate
    cfg.train.lr *= 0.5
    logging.info("Reduced learning rate to {}".format(cfg.train.lr))
    sys.stdout.flush()
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.train.lr

    return optimizer


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    predicted = predicted.detach().argmax(1)
    agreeing = (predicted == true)
    return agreeing


def save_checkpoint(epoch, model, optimizer, model_kwargs, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_kwargs': model_kwargs,
    }
    time.sleep(10)
    torch.save(state, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='msvd_qa_DualVGR.yml', type=str)
    parser.add_argument('--alpha', dest='alpha', help='optional loss parameter', default=1, type=float)
    parser.add_argument('--beta', dest='beta', help='optional loss parameter', default=1e-8, type=float)
    parser.add_argument('--unit_layers', dest='unit_layers', help='unit layers', default=1, type=int)

    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    assert cfg.dataset.name in ['svqa', 'msrvtt-qa', 'msvd-qa']
    # check if the data folder exists
    assert os.path.exists(cfg.dataset.data_dir)


    if not cfg.multi_gpus:
        torch.cuda.set_device(cfg.gpu_id)
    # make logging.info display into both shell and file
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name) # 保存的路径
    if not os.path.exists(cfg.dataset.save_dir):
        os.makedirs(cfg.dataset.save_dir) # 如果没有路径，则创建
    else:
        assert os.path.isdir(cfg.dataset.save_dir) # 检测路径是否为目录
    log_file = os.path.join(cfg.dataset.save_dir, "log")
    if not cfg.train.restore and not os.path.exists(log_file):
        os.mkdir(log_file)
    else:
        assert os.path.isdir(log_file)

    fileHandler = logging.FileHandler(os.path.join(log_file, lctime+cfg.model_type+'_stdout.log'), 'w+')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)   
    # setting: loss + unit_layers
    cfg.alpha = args.alpha
    cfg.beta = args.beta
    cfg.unit_layers = args.unit_layers 
    # args display
    for k, v in vars(cfg).items():
        logging.info(k + ':' + str(v))
    # concat absolute path of input files
    cfg.dataset.appearance_feat = '{}_appearance_feat.h5'
    cfg.dataset.motion_feat = '{}_motion_feat.h5' #
    cfg.dataset.vocab_json = '{}_vocab.json' # vocab-index
    cfg.dataset.train_question_pt = '{}_train_questions.pt' # GloVe
    cfg.dataset.val_question_pt = '{}_val_questions.pt' # GloVe
    cfg.dataset.train_question_pt = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.train_question_pt.format(cfg.dataset.name))
    cfg.dataset.val_question_pt = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.val_question_pt.format(cfg.dataset.name))
    cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name))
    cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name))
    cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name))

    # set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    train(cfg)


if __name__ == '__main__':
    main()
