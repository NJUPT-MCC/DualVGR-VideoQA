import argparse
import numpy as np
import os

from datautils import svqa
from datautils import msrvtt_qa
from datautils import msvd_qa

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='svqa', choices=['msrvtt-qa', 'msvd-qa', 'svqa'], type=str)
    parser.add_argument('--answer_top', default=4000, type=int)
    parser.add_argument('--glove_pt',
                        help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--output_pt', type=str, default='data/{}/{}_{}_questions.pt')
    parser.add_argument('--vocab_json', type=str, default='data/{}/{}_vocab.json')
    parser.add_argument('--mode', choices=['train', 'val', 'test'])
    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.dataset == 'msrvtt-qa':
        args.annotation_file = '/home/WangJY/Jianyu_wang/MSRVTT-QA/{}_qa.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        msrvtt_qa.process_questions(args)
    elif args.dataset == 'msvd-qa':
        args.annotation_file = '/home/WangJY/Jianyu_wang/MSVD-QA/{}_qa.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        msvd_qa.process_questions(args)
    elif args.dataset == 'svqa':
        args.annotation_file = '/home/WangJY/Jianyu_wang/SVQA/{}_qa.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        svqa.process_questions(args)


