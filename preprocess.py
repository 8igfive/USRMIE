import os
import argparse
import json
import math
import numpy as np
from tqdm import tqdm
from typing import Dict, List, cast, Union
from src.tool.prior_calculation import formalizers, calculators, metrics

DUMP_DIR = r'dump/preprocess'
DUMP_CORPUS_NAME = 'corpus'
DUMP_QAC_NAME = 'qac.json'

CORPUS2TASK = {
    'SemEval2016': 'QA',
    'SemEval2017': 'QA',
    'WikiQA': 'QA',
    'TrecQA': 'QA',
    'WikiPassageQA': 'PR',
    'WSD': 'WSD'
}
CORPUS2PATH = { # 0 for dev, 1 for test, 2 for train |WSD is special|
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
    'WSD': [
        r'data/wsd/querys.json',                    # query
        r'data/wsd/gloss.json',                     # gloss
        r'data/wsd/evaluation/ALL/ALL_data.json'    # data
    ]
}
TASK2FORMALIZER = {
    'QA': formalizers.QAFormalizer,
    'PR': formalizers.PRFormalizer,
    'WSD': formalizers.WSDFormalizer
}

METHODS2CALCULATORS = {
    'USE': calculators.USECalculator,
    'SBert': calculators.SBertCalculator,
    'BertNSP': calculators.BertNSPCalculator,
    'SimCSE': calculators.SimCSECalculator,
}

PR_METHODS = ['normal', 'max', 'mean']

def formalize(corpus: str, dump_dir: str = DUMP_DIR, dump_corpus: bool = True, dump_querys_and_candidates: bool = True):
    assert corpus in CORPUS2TASK

    task = CORPUS2TASK[corpus]

    formalizer = TASK2FORMALIZER[task](
        CORPUS2PATH[corpus]
    )

    dump_dir = os.path.join(dump_dir, corpus)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir, exist_ok=True)

    if dump_corpus:
        formalizer.dump_corpus(os.path.join(dump_dir, DUMP_CORPUS_NAME))
    
    if dump_querys_and_candidates:
        formalizer.dump_querys_and_candidates(os.path.join(dump_dir, DUMP_QAC_NAME))

def _cal_sdf_qa(calculator: calculators.BaseCalculator, 
                  samples: List[Dict[str, Union[str, List[Dict[str, str]]]]], 
                  whitening: bool = False) -> Dict[str, List[float]]:
    
    if whitening:
        print('prepare bias and W for whitening')
        for sample in tqdm(samples):
            query = [sample['query']]
            candidates = [candidate['content'] for candidate in sample['candidates']]
            sents = query + candidates
            calculator.fit(sents)
    
    priors = {}
    for sample in tqdm(samples):
        query = [sample['query']]
        candidates = [candidate['content'] for candidate in sample['candidates']]
        scores = calculator.calculate_score(query, candidates, whitening)
        priors[sample['id']] = [score.item() for score in scores[0]]
    return priors

def _cal_sdf_pr(calculator: calculators.BaseCalculator, 
                  samples: List[Dict[str, Union[str, List[Dict[str, str]]]]],
                  pr_method: str = 'normal', max_per_time: int = 20, 
                  whitening: bool = False) -> Dict[str, List[float]]:
    assert pr_method in PR_METHODS

    if whitening:
        print('prepare bias and W for whitening')
        for sample in tqdm(samples):
            query = [sample['query']]
            candidates = [candidate['content'] for candidate in sample['candidates']]
            if pr_method == 'normal':
                sents = query + candidates
            else:
                split_candidates = []
                for candidate in candidates:
                    sentences = formalizers.PRFormalizer.passage_split(candidate)
                    split_candidates += sentences if sentences else [candidate]
                sents = query + split_candidates
            for i in range(math.ceil(len(sents) / max_per_time)):
                calculator.fit(sents[i * max_per_time: (i + 1) * max_per_time])

    priors = {}
    for sample in tqdm(samples):
        query = [sample['query']]
        candidates = [candidate['content'] for candidate in sample['candidates']]
        if pr_method == 'normal':
            scores = []
            for i in range(math.ceil(len(candidates) / max_per_time)): # 解决显存不足问题
                temp_scores = calculator.calculate_score(query, 
                                candidates[i * max_per_time: (i + 1) * max_per_time],
                                whitening)
                scores += [score.item() for score in temp_scores[0]]
        else:
            scores = []
            for candidate in candidates:
                sentences = formalizers.PRFormalizer.passage_split(candidate)
                sentences = sentences if sentences else [candidate]
                sent_scores = []
                for i in range(math.ceil(len(sentences) / max_per_time)): # 解决显存不足问题
                    temp_sent_scores = calculator.calculate_score(query, 
                                        sentences[i * max_per_time: (i + 1) * max_per_time],
                                        whitening)
                    sent_scores += [score.item() for score in temp_sent_scores[0]]
                if pr_method == 'max':
                    score = max(sent_scores)
                elif pr_method == 'mean':
                    score = sum(sent_scores) / len(sent_scores)
                scores.append(score)
        priors[sample['id']] = scores
    return priors

TASK2CAL_FUNC = {
    'QA': _cal_sdf_qa,
    'PR': _cal_sdf_pr,
    'WSD': _cal_sdf_qa
}

def cal_source_domain_function(corpus: str, method: str, dump_dir: str = DUMP_DIR, 
              model_path: str = None, pr_method: str = 'normal', 
              whitening: bool = False, suffix: str = ''):
    assert corpus in CORPUS2PATH and method in METHODS2CALCULATORS, f'Only support corpus in {list(METHODS2CALCULATORS)}'
    assert model_path, 'Please pass a model path.'
    if pr_method:
        assert pr_method in PR_METHODS

    task = CORPUS2TASK[corpus]

    calculator = cast(calculators.BaseCalculator, METHODS2CALCULATORS[method](
        model_path=model_path
    ))

    with open(os.path.join(DUMP_DIR, corpus, DUMP_QAC_NAME)) as fi:
        samples = json.load(fi)
    
    cal_func = TASK2CAL_FUNC[task]
    cal_params = {
        'calculator': calculator,
        'samples': samples,
        'whitening': whitening
    }
    if pr_method:
        cal_params['pr_method'] = pr_method
    priors = cal_func(**cal_params)
    
    dump_dir = os.path.join(dump_dir, corpus, 'priors')
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir, exist_ok=True)
    
    if pr_method:
        file_name = f'{method}_{pr_method}'
    else:
        file_name = f'{method}'
    if suffix:
        file_name = f'{file_name}_{suffix}'
    file_name = f'{file_name}.json'

    with open(os.path.join(dump_dir, file_name), 'w', encoding='utf8') as fo:
        json.dump(priors, fo, ensure_ascii=False, indent=4)

def cal_metrics(metric: str, corpus: str, preds_path: str, cal_range: str, dump_dir: str = DUMP_DIR):
    assert metric in metrics.AVAILABLE_METRICS
    assert corpus in CORPUS2PATH

    cal_indices = []
    for i, data_type in enumerate(['dev', 'test', 'train']):
        if data_type in cal_range:
            path = CORPUS2PATH[corpus][i]
            with open(path, 'r', encoding='utf8') as fi:
                samples = json.load(fi)
            cal_indices += [sample['id'] for sample in samples]
    cal_indices = set(cal_indices)

    corpus_qac_path = os.path.join(DUMP_DIR, corpus, DUMP_QAC_NAME)
    with open(corpus_qac_path, 'r', encoding='utf8') as fi:
        samples = json.load(fi)
    labels = {sample['id']: [int(candidate['label']) for candidate in sample['candidates']] 
                                        for sample in samples if sample['id'] in cal_indices}

    with open(preds_path, 'r', encoding='utf8') as fi:
        preds = json.load(fi)

    metrics_calculator = metrics.Metrics()

    preds_name = os.path.split(os.path.splitext(preds_path)[0])[-1]
    metrics_result = metrics_calculator.cal_metric(metric, labels, preds)

    print('=' * 20)
    print(f'Metric: {metric}')
    print(f'Corpus: {corpus}')
    print(f'Preds: {preds_name}')
    print(f'Result: {metrics_result}')
    print('=' * 20)

    dump_path = os.path.join(dump_dir, corpus, metric)
    if not os.path.exists(dump_path):
        with open(dump_path, 'w', encoding='utf8') as fo:
            fo.write('Preds\tResult\n')
    with open(dump_path, 'a', encoding='utf8') as fo:
        fo.write(f'{os.path.split(os.path.splitext(preds_path)[0])[-1]}\t{metrics_result}\n')

def generate_DIY_label(base_p_sum: float, corpus: str, preds_path: str, dump_dir: str = DUMP_DIR):
    assert corpus in CORPUS2PATH

    with open(preds_path, 'r', encoding='utf8') as fi:
        preds = cast(dict, json.load(fi))
    
    res_preds = {}
    for id, pred in tqdm(preds.items()):
        num_cand = len(pred)
        step = (1 - base_p_sum) / (num_cand * (num_cand - 1) / 2)
        base_p = base_p_sum / num_cand

        new_pred = [0] * num_cand
        for seq, index in enumerate(np.argsort(pred)):
            new_pred[index] = base_p + step * seq
        res_preds[id] = new_pred

    dump_dir = os.path.join(dump_dir, corpus, 'priors')
    preds_name = f'{os.path.split(preds_path)[-1][:-5]}_DIY_{base_p_sum}.json'

    with open(os.path.join(dump_dir, preds_name), 'w', encoding='utf8') as fo:
        json.dump(res_preds, fo, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fc', '--formalize_corpus', type=str, 
                        choices=list(CORPUS2PATH), default=None, help='the corpus to be formalized')
    parser.add_argument('-dd', '--dump_dir', type=str, 
                        default=DUMP_DIR, help='directory to dump results.')
    parser.add_argument('-dc', '--dump_corpus', 
                        action='store_true', help='whether to dump corpus')
    parser.add_argument('-dqac', '--dump_querys_and_candidates', 
                        action='store_true', help='whether to dump querys and candidates')
    parser.add_argument('-cm', '--cal_method', type=str, 
                        choices=list(METHODS2CALCULATORS), default=None, help='the method to calculate prior')
    parser.add_argument('-c', '--corpus', type=str,
                        choices=list(CORPUS2PATH), default='SemEval2016', help='the corpus to be calculated prior')
    parser.add_argument('-mp', '--model_path', type=str, default=None, help='the path for model to use')
    parser.add_argument('-prm', '--pr_method', type=str, default=None,
                        choices=PR_METHODS, help='the passage ranking method for calculate prior')
    parser.add_argument('-wh', '--whitening', action='store_true', help='whether activate whitening')
    parser.add_argument('-cmx', '--cal_metrics', type=str,
                        choices=metrics.AVAILABLE_METRICS, default=None, help='the metrics to be calculated')
    parser.add_argument('-pp', '--preds_path', type=str, default='dump/preprocess/SemEval2016/priors/TFIDF_0.json',
                        help='the path that contains preds')    
    parser.add_argument('-cr', '--cal_range', type=str, default='testdevtrain', help='the range to cal metrics')             
    parser.add_argument('-sf', '--suffix', type=str, default='', help='suffix for dump file')
    parser.add_argument('-gdl', '--generate_DIY_label', 
                        action='store_true', help='whether to generate DIY label')
    parser.add_argument('-bps', '--base_p_sum', type=float, default=0.5, 
                        help='the sum of total base_p')
    


    args = parser.parse_args()

    if args.formalize_corpus:
        formalize(args.formalize_corpus, args.dump_dir, 
                  args.dump_corpus, args.dump_querys_and_candidates)

    if args.cal_method:
        cal_source_domain_function(args.corpus, args.cal_method, 
                  args.dump_dir, args.model_path, args.pr_method, 
                  args.whitening, args.suffix)

    if args.cal_metrics:
        cal_metrics(args.cal_metrics, args.corpus, args.preds_path, args.cal_range, 
                    args.dump_dir)

    if args.generate_DIY_label:
        generate_DIY_label(args.base_p_sum, args.corpus, args.preds_path, 
                    args.dump_dir)


if __name__ == '__main__':

    main()
