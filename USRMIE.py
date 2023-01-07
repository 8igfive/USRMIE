import argparse
import logging
import torch
import numpy as np
import random
import yaml
import os
import shutil
import transformers
import torch.distributed as dist
from src.train.trainer import Trainer

from src.model import MODELS

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A pytorch implementation of USRMIE.')
    parser.add_argument('-db', '--debug', action='store_true', help='whethet to activate debug mode')
    parser.add_argument('-t', '--task', type=str, help='the task to execute')
    parser.add_argument('-cp', '--config_path', type=str, help='path of config file')
    parser.add_argument('-dtr', '--do_train', action='store_true', help='whether to train model')
    parser.add_argument('-dte', '--do_test', action='store_true', help='whether to test model')
    parser.add_argument('-ct', '--continue_training', action='store_true', help='whether to continue training')
    parser.add_argument('-lm', '--load_model', type=str, default=None, help='model path for continuing training')
    parser.add_argument('-sd', '--save_dir', type=str, default='dump/checkpoint', help='directory to hold dump')
    parser.add_argument('-pe', '--print_every', type=int, default=100, help='the interval of output log')
    parser.add_argument('-g', '--gpus', type=str, default='', help='gpus to use')
    parser.add_argument('-r', '--local_rank', type=int, default=0, help='local_rank of distributed training')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed to use')
    parser.add_argument('-wf', '--writer_folder', type=str, default='dump/tensorboard', help='folder to contain SummaryWriter result')
    parser.add_argument('-wc', '--writer_comment', type=str, default='', help='comment in SummaryWriter')
    args = parser.parse_args()

    if args.debug:
        print(args)

    # log part
    LOG_FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=(logging.DEBUG if args.debug else logging.INFO), format=LOG_FORMAT)
    transformers.logging.set_verbosity_error()

    # random seed part
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger.info(f'manual seed is {args.seed}')

    # load parameters
    logger.info('load config from {}'.format(args.config_path))
    with open(args.config_path, 'r', encoding='utf8') as fin:
        params = yaml.load(fin, Loader=yaml.FullLoader)

    # save config
    args.save_path = os.path.join(args.save_dir, params['train']['name'])
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    shutil.copy(args.config_path, os.path.join(args.save_path, 'config.yaml'))

    # distributed part
    logger.info('local_rank: {}, gpus: {}'.format(args.local_rank, args.gpus))
    if len(args.gpus) > 0:
        logger.info('Set CUDA_VISIBLE_DEVICES as %s' % args.gpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if len(args.gpus.split(',')) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method='env://',
                                        rank=args.local_rank, world_size=len(args.gpus.split(',')))

    trainer = Trainer(args, params)
    logger.info('build trainer successfully')
    if args.do_train:
        logger.info('start training')
        try:
            trainer.train()
        except Exception:
            logger.error('Train failed', exc_info=True)

    if args.do_test and args.local_rank == 0:
        logger.info('test after training')
        try:
            trainer.test_after_train()
        except Exception:
            logger.error('Test failed', exc_info=True)
