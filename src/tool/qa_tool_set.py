import logging
import math
import torch
import pdb

logger = logging.getLogger(__name__)

def modify_pred_nop(pred: float) -> float:
    return pred

def modify_pred_exp(pred: float) -> float:
    return math.exp(pred)

def test_iterator(model, data, dataset, args, params):
    tokenizer = dataset.tokenizer

    uid = data['id']

    cand_use_method = params['data']['cand_use_method']
    scores = []
    labels = []
    for i, cand in enumerate(data['candidates']):        
        qncs = dataset.generate_bertinput(data, cand)
        if int(cand['label']) == 0:
            labels.append((cand['cid'], 0))
        else:
            labels.append((cand['cid'], 1))
        
        inputs = tokenizer(qncs, padding=True, truncation=True, 
                    max_length=params['data']['max_length'], return_tensors='pt')
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        attention_mask = inputs['attention_mask']
        if len(args.gpus) > 0:
            input_ids = input_ids.cuda(args.local_rank)
            token_type_ids = token_type_ids.cuda(args.local_rank)
            attention_mask = attention_mask.cuda(args.local_rank)
        with torch.no_grad():
            cand_scores = model((input_ids, token_type_ids, attention_mask))
            if cand_use_method == 'normal':
                assert cand_scores.shape[0] == 1
                scores.append(cand_scores[0])
            else:
                if cand_use_method == 'max':
                    scores.append(torch.max(cand_scores))
                elif cand_use_method == 'mean':
                    scores.append(torch.mean(cand_scores))
    scores = torch.stack(scores, dim=0)
    scores = torch.softmax(scores, 0)

    priors =  dataset.prior_preds[uid]
    priors = torch.Tensor(priors)
    preds = dataset.preds[uid]
    preds = torch.Tensor(preds)
    if len(args.gpus) > 0:
        priors = priors.cuda(args.local_rank)
        preds = preds.cuda(args.local_rank)
    
    if scores.min() < 0:
        scores -= scores.min()  # 非负化
    if params['train']['preds_update_base']:
        scores = torch.pow(scores, params['train']['preds_update_score_power'])
        preds_update_base = params['train']['preds_update_base']
        if preds_update_base == 'priors':
            update_base = torch.pow(priors, params['train']['preds_update_base_power'])
        elif preds_update_base == 'preds':
            update_base = torch.pow(preds, params['train']['preds_update_base_power'])
        else:
            logger.error('Wrong preds_update_base')
        scores = scores * update_base
        # scores = pred
    scores = scores / torch.sum(scores)
    scores = scores.tolist()    # scores = [round(s,4) for s in scores]
    # result: [(cid, score, label), ...]
    result = [(v[0], s, v[1]) for s, v in zip(scores, labels)]
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    # sorted_result = sorted(result,key = lambda x:x[2])    # ? 这个排序是有问题的
    result = [(v, round(s, 4), l) for v, s, l in result]

    used_metrics = params['train']['metric']
    metrics_all = params['train']['metrics_all']
    met = {}

    if used_metrics == 'MAP' or metrics_all:
        tot, n_right = 0.0, 0.0
        ar = 0
        for index, res in enumerate(sorted_result):
            if res[2] == 1:
                n_right += 1.0
                tot += n_right / (index + 1.0)
                if ar == 0:
                    ar = 1 / (index + 1.0)
        ap = 0 if n_right == 0 else tot / n_right
        if metrics_all:
            met['MAP'] = ap
        else:
            met = ap
        # return result, ap
    if used_metrics == 'MRR' or metrics_all:
        ar = 0
        for index, res in enumerate(sorted_result):
            if res[2] == 1:
                if ar == 0:
                    ar = 1 / (index + 1.0)
                break
        if metrics_all:
            met['MRR'] = ar
        else:
            met = ar
        # return result, ar
    if used_metrics == 'P5':
        tot, n_right = 0.0, 0.0
        for index, res in enumerate(sorted_result[:5]):
            if res[2] == 1:
                n_right += 1.0
        # ap = 0 if n_right == 0 else tot / n_right
        met = n_right / 5
        # return result, n_right / 5

    return result, met

def test(model, datas, dataset, args, params):
    model = model.eval()
    results = []
    metrics_all = params['train']['metrics_all']
    if metrics_all:
        metrics = {}
    else:
        metrics = 0
    # query_aps = {}
    for data in datas:
        # print(data)
        qname = data['id']
        res, metric = test_iterator(
            model, data, dataset, args, params)
        # [(id, [(cid, score, label), ...], ap), ...]
        results.append((qname, res, metric))
        if metrics_all:
            for key, value in metric.items():
                metrics[key] = metrics.get(key, 0) + value
        else:
            metrics += metric
    if metrics_all:
        for key, value in metrics.items():
            metrics[key] = value / len(datas)
    else:
        metrics /= len(datas)
    return metrics, results

def write_result(resultname, results):
    with open(resultname, 'w') as fo:
        for ele in results:
            fo.write(f'{ele[0]}\t{ele[1]}\t{ele[2]}\n')

def get_new_preds(model, dataset, args, params):
    model.eval()
    new_preds = dict()
    for data in dataset.data:
        res, _ = test_iterator(model, data, dataset, args, params)
        temp_score = [result[1] for result in res]
        tot = sum(temp_score)
        temp_score = [score / tot for score in temp_score]
        new_preds[data['id']] = temp_score
    return new_preds

TOOL_SET = {
    'modify_pred': {
        'nop': modify_pred_nop,
        'exp': modify_pred_exp
    },
    'get_new_preds': get_new_preds, 
    'test_iterator': test_iterator,
    'test': test,
    'write_result': write_result
}