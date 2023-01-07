import logging
from ..model import MODELS
from ..data import DATASETS
from ..tool import TOOL_SETS, DEFAULT_CONFIG
from ..tool.prior_calculation.metrics import cal_metrics4training
from . import OPTIMIZERS
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
from typing import cast, List
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, args, params):
        self.args = args
        self.params = params

        self.best_model_path = None
        self.task = self.args.task
        self.used_metrics = self.params['train'].get('metric', 
                                DEFAULT_CONFIG['train']['metric'])

        # model part
        self.model = MODELS[params['model']['model_type']](params['model'])
        if args.continue_training and args.load_model is not None:
            self.load_model(args.load_model)
            # FIXME: different from original implement, may contain bugs
        if len(args.gpus) > 0:
            self.model = self.model.cuda(args.local_rank)
        if len(args.gpus.split(',')) > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[args.local_rank], output_device=args.local_rank
            )
            logger.info('[DDP] Use %d gpus for training!' % len(args.gpus.split(',')))
        
        self.dataset = DATASETS[self.task](params)

        # optimizer and scheduler
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = OPTIMIZERS[params['train']['optimizer']['type']](
            self.parameters,
            **dict(
                filter(
                    lambda x: x[0] != 'type',
                    params['train']['optimizer'].items()
                )
            )
        )

        # writer part
        global writer
        writer = SummaryWriter(args.writer_folder, comment=args.writer_comment)

    def train(self):

        self._load_component()

        self._cal_update_steps()

        self._init_statistics()

        if self.args.local_rank == 0:
            confirm = self._cal_confirm(self.preds)
            logger.info('confirm:%s/%s' % (confirm, len(self.preds)))
            logger.info('M_step')

        # without source
        self._wo_source() # with be without source depending on the params

        for epoch in range(self.params['train']['epoch']):
            self.tot_loss = 0
            self.loss_time = 0
            self.tot_pos_score = 0
            self.tot_neg_score = 0
            for i in range(self.update_steps):
                # M_step
                self._train_one_step(i)
            
            self.stop += 1

            self._post_train(epoch)

            if self._early_stop():
                self._epoch_check_record(epoch)
                return
            
            # E_step
            self.E_step(epoch)

            self._epoch_check_record(epoch)

    def E_step(self, epoch):
        
        wo_estep = self.params['train'].get('wo_estep', 
                        DEFAULT_CONFIG['train']['wo_estep'])

        new_preds = self.get_new_preds(self.model, self.dataset, self.args, self.params)
        if self.args.local_rank == 0:
            logger.info('E_step')
            tot_dis = self._cal_difference(ori_preds=self.preds, new_preds=new_preds)
            writer.add_scalar('tot_update_dis', tot_dis, epoch)
            logger.info('different after training: {}'.format(tot_dis))

        if not wo_estep:
            logger.info('preds not updated for wo_estep')
            self.preds.update(new_preds)
            
        if self.args.local_rank == 0:
            confirm = self._cal_confirm(self.preds)
            writer.add_scalar('confirm', confirm, epoch)
            logger.info('confirm:%s/%s' % (confirm, len(self.preds)))

            logger.info('complete E_step')    

    def M_step(self, batch):
        loss = 0
        model = self.model.train()

        if len(self.args.gpus) > 0:
            for key, input in batch.items():
                if isinstance(input, torch.Tensor):
                    batch[key] = input.cuda(self.args.local_rank)

        input_ids = cast(torch.Tensor, batch['input_ids'])              # shape of input_size, seq_len
        token_type_ids = cast(torch.Tensor, batch['token_type_ids'])    # shape of input_size, seq_len          
        attention_mask = cast(torch.Tensor, batch['attention_mask'])    # shape of input_size, seq_len
        cands_end = cast(List[int], batch['cands_end'])                 # len = cand_size + 1
        labels = cast(torch.Tensor, batch['labels'])                    # shape of cand_size
        priors = cast(torch.Tensor, batch['priors'])                    # shape of cand_size
        bias = cast(torch.Tensor, batch['bias'])                        # shape of query_size
        samples_pos = cast(List[List[int]], batch['samples_pos'])       # size of query_size, 3
        pos_preds_sum = batch['pos_preds_sum']                          # sum of positive preds
        neg_count = batch['neg_count']                                  # negtive counts

        input_scores = model((input_ids, token_type_ids, attention_mask))   # shape of input_size

        cand_use_method = self.params['data'].get('cand_use_method', 
                                DEFAULT_CONFIG['data']['cand_use_method'])
        with_priors = self.params['train'].get('with_priors',
                                DEFAULT_CONFIG['train']['with_priors'])

        if cand_use_method == 'normal':
            cand_scores = input_scores
        else:
            if cand_use_method == 'max':
                cand_use_fn = torch.max
            elif cand_use_method == 'mean':
                cand_use_fn = torch.mean
            cand_scores = torch.stack(
                [cand_use_fn(input_scores[cands_end[i]: cands_end[i + 1]]) for i in range(len(labels))]
            )
        if with_priors:
            if priors.dtype != cand_scores.dtype:
                priors = priors.to(cand_scores.dtype).to(cand_scores.device)

            cand_scores += torch.log(priors)

        loss_fn = self.params['train'].get('loss', DEFAULT_CONFIG['train']['loss'])
        if loss_fn['type'] == 'nce':
            return self._NCELoss(cand_scores, bias, labels, pos_preds_sum, neg_count, samples_pos)
        elif loss_fn['type'] == 'infonce':
            return self._InfoNCELoss(cand_scores, labels, pos_preds_sum, samples_pos)
        
    def _NCELoss(self, cand_scores, bias, labels, pos_preds_sum, neg_count, samples_pos):
        bias = torch.cat(
            [torch.stack([bias[i]] * (samples_pos[i][2] - samples_pos[i][0]), dim=0)  
                for i in range(len(samples_pos))],
            dim=0
        )   # shape of cand_size

        delta_score = cand_scores - bias
        pos_label = torch.cat(
            [torch.tensor([1] * (sample_pos[1] - sample_pos[0]) + [-1] * (sample_pos[2] - sample_pos[1])) \
                for sample_pos in samples_pos],
            dim=0
        ).to(cand_scores.device)
        delta_score = delta_score * pos_label
        logsig_score = F.logsigmoid(delta_score)

        k = self.params['data'].get('k', DEFAULT_CONFIG['data']['k'])
        probs = []  # list of tensors
        for sample_pos in samples_pos:
            probs += [
                labels[sample_pos[0]: sample_pos[1]] / pos_preds_sum,
                torch.ones(sample_pos[2] - sample_pos[1], 
                    device=labels.device, dtype=labels.dtype) * (k / neg_count)
            ]
        probs = torch.cat(probs, dim=0).to(labels.device).to(labels.dtype)
        prob_logsig_score = logsig_score * probs

        pos_score = torch.cat(
            [prob_logsig_score[sample_pos[0]: sample_pos[1]] for sample_pos in samples_pos]
            , dim=0
        ).sum()
        neg_score = torch.cat(
            [prob_logsig_score[sample_pos[1]: sample_pos[2]] for sample_pos in samples_pos]
            , dim=0
        ).sum()

        loss = -(prob_logsig_score).sum()
        
        return loss, (pos_score, neg_score)

    def _InfoNCELoss(self, cand_scores, labels, pos_preds_sum, samples_pos):
        loss_fn = nn.CrossEntropyLoss()
        loss = 0

        temperature = self.params['train'].get('loss', 
                    DEFAULT_CONFIG['train']['loss'])['temperature']

        for sample_pos in samples_pos:
            for pos_position in range(sample_pos[0], sample_pos[1]):
                cal_sample = torch.cat(
                    [cand_scores[pos_position].view(-1), 
                        cand_scores[sample_pos[1]: sample_pos[2]]],
                    dim=0
                ) / temperature
                prob = labels[pos_position] / pos_preds_sum
                loss += prob * loss_fn(cal_sample[None, : ], 
                                torch.zeros(1, dtype=torch.long, device=cal_sample.device))
        
        pos_score = torch.cat(
            [cand_scores[sample_pos[0]: sample_pos[1]] for sample_pos in samples_pos]
            , dim=0
        ).sum()
        neg_score = torch.cat(
            [cand_scores[sample_pos[1]: sample_pos[2]] for sample_pos in samples_pos]
            , dim=0
        ).sum()

        return loss, (pos_score, neg_score)

    def _load_component(self):
        # TOOLS
        self.get_new_preds = TOOL_SETS[self.task]['get_new_preds']
        self.test = TOOL_SETS[self.task]['test']
        self.write_result = TOOL_SETS[self.task]['write_result']
        # DATA
        self.preds = self.dataset.preds
        self.data = self.dataset.data
        self.dev_data= self.dataset.dev_data
        self.test_data = self.dataset.test_data

    def _cal_update_steps(self):
        update_steps = self.params['train'].get('update_steps', None)
        self.update_steps = update_steps if update_steps else len(self.dataset)
        
        world_size = len(self.args.gpus.split(',')) if self.args.gpus else 1
        batch_size = self.params['data'].get('mini_batch',  
                                    DEFAULT_CONFIG['data']['mini_batch'])
        self.update_steps = int(self.update_steps / world_size / batch_size)

        logger.info(f'Set update_steps={self.update_steps}')

    def _init_statistics(self):

        self.max_dev_metrics = 0
        self.stop = 0

    def _train_one_step(self, step):
        batch = self.dataset.generate_batch()
        loss, (pos_score, neg_score) = self.M_step(batch)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.optimizer.step()
        
        # collect loss and log
        l = loss.item()
        self.tot_loss += l
        self.loss_time += 1
        self.tot_pos_score += pos_score.item()
        self.tot_neg_score += neg_score.item()
        
        # log
        if self.args.local_rank == 0 and step % self.args.print_every == 0:
            logger.info('M_step: {}/{}({:.4f}%) loss={:.4f}, p_score={:.4f}, n_score={:.4f}, avg_loss={:.4f}, avg_p_score={:.4f}, avg_n_score={:.4f}'.format(
                step, self.update_steps, step / self.update_steps * 100, 
                l, pos_score, neg_score, self.tot_loss / self.loss_time, self.tot_pos_score / self.loss_time, self.tot_neg_score / self.loss_time
            ))

    def _post_train(self, epoch):
        if self.task == 'wsd':
            self._post_train_wsd(epoch)
        elif self.task == 'qa':
            self._post_train_qa(epoch)

    def _post_train_wsd(self, epoch):
        
        _, dev_F1, results = self.test(self.model, self.test_data, self.dataset, self.args, self.params)

        # epoch save
        if self.args.local_rank == 0:
            logger.info('complete M_step')

            writer.add_scalars('loss', {'loss': self.tot_loss / self.loss_time}, epoch)
            writer.add_scalars('scores', {'pos_score': self.tot_pos_score / self.loss_time, 'neg_score': self.tot_neg_score / self.loss_time},
                            epoch)
            writer.add_scalar('dev_F1', dev_F1, epoch)

            self.write_result(os.path.join(self.args.save_path, 'epoch' + str(epoch) + 'all.test.res'), results)
            with open(os.path.join(self.args.save_path, 'epoch' + str(epoch) + 'all.preds.txt'), 'w', encoding='utf8') as fout:
                for key in self.preds:
                    fout.write(key + '\t' + str(self.preds[key]) + '\n')
            logger.info('epoch: %s, avg_loss:%s' % (epoch, self.tot_loss / self.loss_time))
            logger.info('dev_F1: %s/%s' % (dev_F1, self.max_dev_metrics))
        # best save
        if dev_F1 > self.max_dev_metrics:
            self.stop = 0
            if self.args.local_rank == 0:
                if os.path.exists(os.path.join(self.args.save_path, str(self.max_dev_metrics) + 'all.model.pkl')):
                    os.remove(os.path.join(self.args.save_path, str(self.max_dev_metrics) + 'all.test.res'))
                    os.remove(os.path.join(self.args.save_path, str(self.max_dev_metrics) + 'all.model.pkl'))
                    os.remove(os.path.join(self.args.save_path, str(self.max_dev_metrics) + 'all.preds.txt'))
                self.write_result(os.path.join(self.args.save_path, str(dev_F1) + 'all.test.res'), results)
                with open(os.path.join(self.args.save_path, str(dev_F1) + 'all.preds.txt'), 'w', encoding='utf8') as fout:
                    for key in self.preds:
                        fout.write(key + '\t' + str(self.preds[key]) + '\n')
                check_point = {}
                check_point['model_dict'] = self.model.state_dict()
                self.best_model_path = os.path.join(self.args.save_path, str(dev_F1) + 'all.model.pkl')
                torch.save(check_point, self.best_model_path)
        
        self.max_dev_metrics = max(self.max_dev_metrics, dev_F1)
        self.loss_time = 0
        self.tot_loss = 0

    def _post_train_qa(self, epoch):
        
        self.params['train']['metrics_all'] = True
        _, results = self.test(self.model, self.test_data, self.dataset, self.args, self.params)
        self.params['train']['metrics_all'] = False
        dev_metrics, _ = self.test(self.model, self.dev_data, self.dataset, self.args, self.params)
        dev_metrics = round(dev_metrics, 4)

        # epoch save
        if self.args.local_rank == 0:
            logger.info('complete M_step')

            loss_time = max(self.loss_time,1)

            writer.add_scalar('loss', self.tot_loss / loss_time, epoch)
            writer.add_scalars('scores', {'pos_score': self.tot_pos_score / loss_time, 'neg_score': self.tot_neg_score / loss_time},
                            epoch)
            writer.add_scalar(f'dev_{self.used_metrics}', dev_metrics, epoch)

            self.write_result(os.path.join(self.args.save_path, 'epoch' + str(epoch) + 'all.test.res'), results)
            with open(os.path.join(self.args.save_path, 'epoch' + str(epoch) + 'all.preds.txt'), 'w', encoding='utf8') as fout:
                for key in self.preds:
                    fout.write(key + '\t' + str(self.preds[key]) + '\n')
            logger.info('epoch: %s, avg_loss:%s' % (epoch, self.tot_loss / loss_time))

            logger.info(f'dev_{self.used_metrics}: {dev_metrics}/{self.max_dev_metrics}')
        # best save
        if dev_metrics > self.max_dev_metrics:
            self.stop = 0
            
            if self.args.local_rank == 0:
                if os.path.exists(os.path.join(self.args.save_path, str(self.max_dev_metrics) + 'all.model.pkl')):
                    os.remove(os.path.join(self.args.save_path, str(self.max_dev_metrics) + 'all.test.res'))
                    os.remove(os.path.join(self.args.save_path, str(self.max_dev_metrics) + 'all.model.pkl'))
                    os.remove(os.path.join(self.args.save_path, str(self.max_dev_metrics) + 'all.preds.txt'))
                self.write_result(os.path.join(self.args.save_path, str(dev_metrics) + 'all.test.res'), results)
                with open(os.path.join(self.args.save_path, str(dev_metrics) + 'all.preds.txt'), 'w', encoding='utf8') as fout:
                    for key in self.preds:
                        fout.write(key + '\t' + str(self.preds[key]) + '\n')
                check_point = {}
                check_point['model_dict'] = self.model.state_dict()
                self.best_model_path = os.path.join(self.args.save_path, str(dev_metrics) + 'all.model.pkl')
                torch.save(check_point, self.best_model_path)
        # maintain the best
        self.max_dev_metrics = max(self.max_dev_metrics, dev_metrics)
        self.loss_time = 0
        self.tot_loss = 0

    def _early_stop(self, max_stop = 30):
        if self.stop >= max_stop:
            if self.args.local_rank == 0:
                logger.info('Not improving for a long time, training terminated.')
            return True
        else:
            return False

    def _epoch_check_record(self, epoch):
        if self.args.local_rank == 0:
            with open(os.path.join(self.args.save_path, 'epoch' + str(epoch) + 'samplecount.txt'), 'w', encoding='utf8') as fout:
                for ele in self.test_data:
                    key = ele['id']
                    neg_count = self.dataset.neg_counts[key] if key in self.dataset.neg_counts else []
                    fout.write(key + '\t' + str(neg_count) + '\n')
            
            if self.task == 'wsd':
                metric_dict = {f'max_dev_{self.used_metrics}': self.max_dev_metrics}
            elif self.task == 'qa':
                metric_dict = {f'max_dev_{self.used_metrics}': self.max_dev_metrics}
            
            writer.add_hparams(
                {'dataset': self.params['data']['dataset'], 'lr': self.params['train']['optimizer']['lr'], 
                    'mb': self.params['data']['mini_batch'], 'neg_method': self.params['data']['neg_method'],
                    'k': self.params['data']['k'], 'update_steps': self.update_steps, 
                    'data_type': self.params['data']['data_type'], 'seed': self.args.seed,}, 
                metric_dict
            )

    def test_after_train(self):
        if self.args.local_rank != 0:
            return
        model_path = self.best_model_path if self.best_model_path \
                        else self.args.load_model
        if not model_path:
            logger.info('Lack of model to test')
        else:
            self.load_model(model_path)
            if self.args.task == 'wsd':
                self._test_after_train_wsd()
            elif self.args.task == 'qa':
                self._test_after_train_qa()

    def _test_after_train_wsd(self):
        test = TOOL_SETS['wsd']['test']
        test_all = TOOL_SETS['wsd']['test_all']

        test_data = self.dataset.test_data

        test_F1, dev_F1, results = test(self.model, test_data, self.dataset, self.args, self.params)

        res = test_all(results, self.args)
        res['all'] = test_F1
        
        with open(os.path.join(self.args.save_path, 'test_all.res'), 'w', encoding='utf8') as fo:
            json.dump(res, fo)

        for key, value in res.items():
            logger.info(f'F1 for {key}: {value}%')
        # logger.info('Improvements: '+str(dev_F1-53.9)+'%')

    def _test_after_train_qa(self):
        
        test = TOOL_SETS['qa']['test']
        metrics, _ = test(self.model, self.dataset.test_data, self.dataset, self.args, self.params)
        logger.info(f'{self.used_metrics}: {metrics}')

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_dict'])

    def _cal_confirm(self, preds, threshold = 0.9):
        confirm = 0
        for key in preds:
            lis = preds[key]
            score = max(lis)
            confirm += 1 if score > threshold else 0
        return confirm

    def _cal_difference(self, ori_preds, new_preds, case_study: bool = True):

        if case_study:
            key = list(new_preds)[0]
            new_pred = [round(p, 4) for p in new_preds[key]]
            new_seq = np.argsort(new_pred)[::-1]
            for seq, index in enumerate(new_seq):
                new_pred[index] = (new_pred[index], seq + 1)
            ori_pred = [round(p, 4) for p in ori_preds[key]]
            ori_seq = np.argsort(ori_pred)[::-1]
            for seq, index in enumerate(ori_seq):
                ori_pred[index] = (ori_pred[index], seq + 1)
            logger.info(f'Case study: preds for {key}\nfrom {ori_pred}\n->   {new_pred}')

        tot_dis = 0
        for key, new_pred in new_preds.items():
            new_pred = np.array(new_pred)
            ori_pred = np.array(ori_preds[key])
            if np.argmax(new_pred) != np.argmax(ori_pred):
                tot_dis += 1
        return tot_dis
    
    def _wo_source(self):
        wo_source = self.params['train'].get('wo_source', 
                        DEFAULT_CONFIG['train']['wo_source'])
        if wo_source:
            self.params['train']['wo_estep'] = True
            self.params['train']['preds_update_base'] = ''
