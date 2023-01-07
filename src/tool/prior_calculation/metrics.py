from typing import Dict, List
from src.tool.wsd_tool_set import evaluate
import numpy as np
import json
import os
import pdb

AVAILABLE_METRICS = ['MAP', 'MRR']

class Metrics(object):
    def __init__(self):
        self.available_metrics = AVAILABLE_METRICS

        self.metric2func = {
            'MAP': self._cal_MAP,
            'MRR': self._cal_MRR
        }

    def cal_metric(self, metric: str, labels: Dict[str, List[int]], preds: Dict[str, List[float]]) -> float:
        assert metric in self.available_metrics

        # SemEval2016-test
        # labels = {id: labels4id for id, labels4id in labels.items() if int(id[1:]) >= 318}

        return self.metric2func[metric](labels, preds)

    def _cal_MAP(self, labels: Dict[str, List[int]], preds: Dict[str, List[float]]) -> float:
        
        map = 0
        for id in labels:
            labels4id = labels[id]
            preds4id = preds[id]

            preds_seq = np.argsort(preds4id, )[::-1].tolist()

            ap = 0
            p_count = 0
            for seq, index in enumerate(preds_seq):
                if labels4id[index] == 1:
                    p_count += 1
                    ap += p_count / (seq + 1)
            if p_count > 0:
                ap /= p_count
            map += ap

        map /= len(labels)
        return map

    def _cal_MRR(self, labels: Dict[str, List[int]], preds: Dict[str, List[float]]) -> float:
        
        mrr = 0
        for id in labels:
            labels4id = labels[id]
            preds4id = preds[id]

            preds_seq = np.argsort(preds4id)[::-1].tolist()
            
            rr = 0
            for seq, index in enumerate(preds_seq):
                if labels4id[index] == 1:
                    rr += 1. / (seq + 1)
                    break
            
            mrr += rr
        
        mrr /= len(labels)
        return mrr

# WSD is individually calculated
CORPUS2PATH = { # 0 for dev, 1 for test, 2 for train
    'SemEval2016': [
        r'data/semeval16/semeval16.dev.json',
        r'data/semeval16/semeval16.test.json',
        r'data/semeval16/semeval16.train.json'
    ],
    'SemEval2017': [
        r'data/semeval17/semeval17.dev.json',
        r'data/semeval17/semeval17.test.json',
        r'data/semeval17/semeval17.train.json'
    ],
    'WikiQA': [
        r'data/wikiqa/wikiqa.dev.json',
        r'data/wikiqa/wikiqa.test.json',
        r'data/wikiqa/wikiqa.train.json'
    ],
    'TrecQA': [
        r'data/trecqa/trecqa.dev.json',
        r'data/trecqa/trecqa.test.json',
        r'data/trecqa/trecqa_wf.train.json'
    ],
    'WikiPassageQA': [
        r'data/wikipassageqa/wikipassageqa.dev.json',
        r'data/wikipassageqa/wikipassageqa.test.json',
        r'data/wikipassageqa/wikipassageqa.train.json'
    ],
}
DUMP_DIR = r'dump/cal_prior'
DUMP_QAC_NAME = 'qac.json'

def cal_metrics4training(corpus: str, preds: Dict[str, List[float]], args=None, params=None):

    if corpus == 'WSD':
        return _cmt_wsd(preds, args, params)
    else:
        return _cmt_qa_and_pr(corpus, preds)

def _cmt_qa_and_pr(corpus: str, preds: Dict[str, List[float]]):
    assert corpus in CORPUS2PATH

    corpus_qac_path = os.path.join(DUMP_DIR, corpus, DUMP_QAC_NAME)
    with open(corpus_qac_path, 'r', encoding='utf8') as fi:
        samples = json.load(fi)
    labels = {sample['id']: [int(candidate['label']) for candidate in sample['candidates']] 
                for sample in samples}

    indices = {}
    for i, index_type in enumerate(['dev', 'test', 'train']):
        corpus_path = CORPUS2PATH[corpus][i]
        if not corpus_path:
            indices[index_type] = []
            continue
        with open(corpus_path, 'r', encoding='utf8') as fi:
                temp_samples = json.load(fi)
        indices[index_type] = [sample['id'] for sample in temp_samples]
    indices['all'] = indices['dev'] + indices['test'] + indices['train']
    for key, index in indices.items():
        indices[key] = set(index)

    metrics_calculator = Metrics()

    res = {}
    for data_type in ['train', 'dev', 'test', 'all']:
        if not indices[data_type]:
            continue
        res[data_type] = {}
        type_labels = {qid: label for qid, label in labels.items() if qid in indices[data_type]}
        for metrics in AVAILABLE_METRICS:
            metrics_res = metrics_calculator.cal_metric(metrics, type_labels, preds)
            res[data_type][metrics] = round(metrics_res, 4) 
    
    return res


def _cmt_wsd(preds: Dict[str, List[float]], args, params):
    pre = 'data/wsd/evaluation'
    with open(params['data']['test_path'], 'r', encoding='utf8') as fi:
        datas = {
            data['id']: data for data in json.load(fi)
        }

    key = 'semeval2007'
    with open(os.path.join(args.save_path, 'ALL.tem.res'), 'w', encoding='utf8') as ALL_out:
        with open(os.path.join(args.save_path, f'{key}.tem.res'), 'w', encoding='utf8') as sub_out:
            for id, pred in preds.items():
                try:
                    neg_candidates = datas[id]['neg_candidates']
                    pred_sense = neg_candidates[np.argmax(pred)]
                    ALL_out.write(f'{id} {pred_sense}\n')
                    if id.startswith(key):
                        sub_out.write(f"{id.replace(f'{key}.', '')} {pred_sense}\n")
                except:
                    pdb.set_trace()
    sub_F1 = evaluate(os.path.join(pre, key, f'{key}.gold.key.txt'), 
                  os.path.join(args.save_path, f'{key}.tem.res'))
    ALL_F1 = evaluate(r'data/wsd/evaluation/ALL/ALL.gold.key.txt',
                  os.path.join(args.save_path, 'ALL.tem.res'))
    res = {
        key: {
            'F1': sub_F1
        },
        'ALL': {
            'F1': ALL_F1
        }
    }
    return res