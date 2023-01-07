from ..tool import TOOL_SETS, DEFAULT_CONFIG
from ._dataset import _Dataset
from typing import cast, List, Tuple
import logging
import random
import torch
import math
import numpy as np
import json
import torch.distributed as dist
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

class Dataset(_Dataset):
    
    def generate_bertinput(self, data, sense):
        '''
        cand_use_method not implemented
        '''
        target_word = data['target_word']
        word = target_word.split('#')[0]

        gloss = word + ":" + " ".join(self.gloss[sense])                       

        context = data['context']
        context = " ".join(context)
        
        return [(context, gloss)]

    def generate_batch(self):
        neg_method = self.params['data'].get('neg_method', 
                            DEFAULT_CONFIG['data']['neg_method'])
        pos_cand_num = self.params['data'].get('pos_cand_num', 
                            DEFAULT_CONFIG['data']['pos_cand_num'])
        k = self.params['data'].get('k', DEFAULT_CONFIG['data']['k'])
        neg_cand_num = self.params['data'].get('neg_cand_num', 
                            DEFAULT_CONFIG['data']['neg_cand_num'])
        pos_main_num = self.params['data'].get('pos_main_num', 
                            DEFAULT_CONFIG['data']['pos_main_num'])
        neg_main_num = self.params['data'].get('neg_main_num', 
                            DEFAULT_CONFIG['data']['neg_main_num'])  
        pos_main_prob = self.params['data'].get('pos_main_prob', 
                            DEFAULT_CONFIG['data']['pos_main_prob'])
        neg_main_prob = self.params['data'].get('neg_main_prob', 
                            DEFAULT_CONFIG['data']['neg_main_prob'])

        # for ablation study
        sample_using_preds = self.params['train'].get('sample_using_preds',
                            DEFAULT_CONFIG['train']['sample_using_preds'])
        sample_uniform = self.params['train'].get('sample_uniform',
                            DEFAULT_CONFIG['train']['sample_uniform'])
        

        inputs = cast(List[Tuple[str]], [])
        cands_end = cast(List[int], [0])
        labels = cast(List[int], [])                        
        priors = cast(List[float], [])                      

        samples_pos = cast(List[List[int]], [])
        bias = cast(List[float], [])
        pos_preds_sum = 0
        neg_count = 0

        select = next(self.index_generater)
        for index in select:
            data = self.data[index]
            qid = data['id']
            target_word = data['target_word']
            word = target_word.split('#')[0]
            candidates = self.querys[target_word]           # senses

            cand_preds = self.preds[qid]
            cand_seq = np.argsort(cand_preds)[::-1]

            position = [len(labels)]
            
            # choice prob part
            if sample_uniform:
                pos_choice_prob = [1 / len(cand_preds)] * len(cand_preds)
                neg_choice_prob = [1 / len(cand_preds)] * len(cand_preds)
            elif sample_using_preds:
                if min(cand_preds) < 0: # ensure >= 0
                    min_cand_preds = min(cand_preds)
                    pos_choice_prob = [pred - min_cand_preds for pred in cand_preds]
                    pos_choice_prob = [pred / sum(pos_choice_prob) for pred in pos_choice_prob]
                else:
                    pos_choice_prob = [pred / sum(cand_preds) for pred in cand_preds]
                pos_choice_prob = [pos_choice_prob[cs] for cs in cand_seq]  # fix a bug for abliation study
                neg_choice_prob = [1 - prob for prob in pos_choice_prob]
                if sum(neg_choice_prob):
                    neg_choice_prob = [prob / sum(neg_choice_prob) for prob in neg_choice_prob]
                else:
                    neg_choice_prob = [1 / len(neg_choice_prob) for i in range(len(neg_choice_prob))]
            else:
                if neg_main_num is None:
                    temp_neg_main_num = max(0, len(candidates) - pos_main_num)
                else:
                    temp_neg_main_num = neg_main_num
                if pos_main_num >= len(candidates):
                    pos_choice_prob = [1 / len(candidates)] * len(candidates)
                else:
                    pos_choice_prob = [pos_main_prob / pos_main_num] * pos_main_num + \
                        [(1 - pos_main_prob) / (len(candidates) - pos_main_num)] * (len(candidates) - pos_main_num)
                if temp_neg_main_num >= len(candidates) or temp_neg_main_num == 0:
                    neg_choice_prob = pos_choice_prob
                else:
                    neg_choice_prob = [(1 - neg_main_prob) / (len(candidates) - temp_neg_main_num)] * \
                                            (len(candidates) - temp_neg_main_num) + \
                                    [neg_main_prob / temp_neg_main_num] * temp_neg_main_num
            
            # pos part
            pos_cand_indices = np.random.choice(cand_seq, size=pos_cand_num, 
                                replace=(pos_cand_num > len([pcb for pcb in pos_choice_prob if pcb > 0])), # fix a bug
                                p=pos_choice_prob)
            for pos_cand_index in pos_cand_indices:
                candidate = candidates[pos_cand_index]  # sense
                cand_pred = cand_preds[pos_cand_index]
                cand_prior = self.priors['{}-{}'.format(qid, candidate)]

                if cand_pred <= 0:
                    continue
                qncs = self.generate_bertinput(data, candidate)
                if not qncs:
                    continue

                inputs += qncs
                cands_end.append(len(inputs))
                labels.append(cand_pred)
                priors.append(cand_prior)
                pos_preds_sum += cand_pred

            if len(labels) == position[-1]:
                continue
            else:
                position.append(len(labels))

            # neg_part
            if neg_method == 'querys':
                neg_cand_num = pos_cand_num * k
                neg_other_num = 0
            if neg_method == 'keys':
                neg_cand_num = 0
                neg_other_num = pos_cand_num * k
            else:   # mix
                neg_other_num = pos_cand_num * k - neg_cand_num
            
            neg_cand_indices = np.random.choice(cand_seq, size=neg_cand_num, 
                                    replace=(neg_cand_num > len([ncb for ncb in neg_choice_prob if ncb > 0])), 
                                    p=neg_choice_prob)
            negs_local = [candidates[index] for index in neg_cand_indices]  # List of senses
            negs_global = np.random.choice(range(len(self.neg_candidates)), 
                                           size=neg_other_num).tolist() # List of (ele['id'], sense)
            negs_global = [self.neg_candidates[neg_index] for neg_index in negs_global]
            negs_sense = negs_local + [neg_sense for _, neg_sense in negs_global]
            negs_id = ['{}-{}'.format(qid, neg_sense) for neg_sense in negs_local] + \
                      ['{}-{}'.format(eid, neg_sense) for eid, neg_sense in negs_global]

            for neg_id, neg_sense in zip(negs_id, negs_sense):
                neg_prior = self.priors[neg_id]

                qncs = self.generate_bertinput(data, neg_sense)
                if not qncs:
                    continue
                    
                inputs += qncs
                cands_end.append(len(inputs))
                labels.append(0)
                priors.append(neg_prior)
                neg_count += 1

                self.neg_counts.setdefault(qid, {})
                self.neg_counts[qid].setdefault(neg_sense, 0)
                self.neg_counts[qid][neg_sense] += 1
            
            if len(labels) == position[-1]:
                continue
            else:
                position.append(len(labels))
            samples_pos.append(position)

            if neg_method == 'querys':
                pn = 1. / (len(candidates))             
            else:
                pn = 1. / (len(self.neg_candidates))
            bias.append(math.log(k * pn))

        max_length = self.params['data'].get('max_length', 
                        DEFAULT_CONFIG['data']['max_length'])
        inputs = self.tokenizer(inputs, padding=True, return_tensors='pt', 
                                max_length = max_length)
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        attention_mask = inputs['attention_mask']
        priors = torch.Tensor(priors)
        labels = torch.Tensor(labels)
        bias = torch.Tensor(bias)

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,

            'cands_end': cands_end,
            'labels': labels,
            'priors': priors,

            'bias': bias,
            'samples_pos': samples_pos,
            'pos_preds_sum': pos_preds_sum,
            'neg_count': neg_count
        }

    def _build_dataset(self):

        path = self.params['data'].get('query_path', DEFAULT_CONFIG['data']['query_path'])
        with open(path, 'r', encoding='utf8') as fin:
            self.querys = json.load(fin)
        path = self.params['data'].get('gloss_path', DEFAULT_CONFIG['data']['gloss_path'])
        with open(path, 'r', encoding='utf8') as fin:
            self.gloss = json.load(fin)
            self.keys = [key for key in self.gloss]
        
        # data part
        self.test_data = []
        self.dev_data = []
        self.train_data = []

        path = self.params['data'].get('test_path', DEFAULT_CONFIG['data']['test_path'])
        if path:
            with open(path, 'r', encoding='utf8') as fin:
                self.test_data = json.load(fin)
        path = self.params['data'].get('dev_path', DEFAULT_CONFIG['data']['dev_path'])
        if path: 
            with open(path, 'r', encoding='utf8') as fin:
                self.dev_data = json.load(fin)
        path = self.params['data'].get('train_path', DEFAULT_CONFIG['data']['train_path'])
        if path:
            with open(self.params['data']['train_path'], 'r', encoding='utf8') as fin:
                self.train_data = json.load(fin)

        data_type = self.params['data'].get('data_type', DEFAULT_CONFIG['data']['data_type'])        
        if data_type:
            self.data = []
            for dtype in ['train', 'dev', 'test']:
                if dtype in data_type:
                    self.data += {'train': self.train_data, 'dev': self.dev_data, 'test': self.test_data}[dtype] #(self.train_data if data_type == 'train' else self.test_data)
                    logger.info(f'Add {dtype} data into data, present data size is {len(self.data)}')
        else:
            self.data = self.train_data + self.dev_data + self.test_data
            logger.info(f'Use test data as data, present data size is {len(self.data)}')

        train_range = set(self.params['data'].get('train_corpus', DEFAULT_CONFIG['data']['train_corpus']))
        if 'all' not in train_range:
            remain_data = []
            for sample in self.data:
                qid = sample['id']
                if qid[:qid.find('.')] in train_range:
                    remain_data.append(sample)
            self.data = remain_data

        path = self.params['data'].get('pred_path', DEFAULT_CONFIG['data']['pred_path'])
        with open(path, 'r', encoding='utf8') as fin:
            self.preds = json.load(fin)
        self.prior_preds = self.preds.copy()
        self.priors = {}                          
        for ele in self.train_data + self.dev_data + self.test_data:
            if ele['id'] not in self.preds:
                continue
            for sense, p in zip(self.querys[ele['target_word']], self.preds[ele['id']]):
                self.priors['{}-{}'.format(ele['id'], sense)] = p
            
        # neg part
        for i, datum in enumerate(self.data):
            target_word = datum['target_word']
            datum['neg_candidates'] = self.querys[target_word]
            self.data[i] = datum
        self.neg_counts = {}

        self.neg_candidates = []
        for ele in self.train_data + self.dev_data + self.test_data:
            if ele['id'] not in self.preds:
                continue
            self.neg_candidates += [(ele['id'], sense) for sense in self.querys[ele['target_word']]]
