import torch
import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def test_iterator(model, data, dataset, args, params):
    tokenizer = dataset.tokenizer
    querys = dataset.querys

    qname = data['target_word']
    target_word = qname.split('#')[0]

    docs = []
    for sense in querys[qname]:
        docs += dataset.generate_bertinput(data, sense)
    inputs = tokenizer(docs, padding=True, truncation=True, 
                    max_length=params['data']['max_length'], return_tensors='pt')
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']
    priors = dataset.prior_preds[data['id']]
    priors = torch.Tensor(priors)
    preds = dataset.preds[data['id']]
    preds = torch.Tensor(preds)
    if len(args.gpus) > 0:
        input_ids = input_ids.cuda(args.local_rank)
        token_type_ids = token_type_ids.cuda(args.local_rank)
        attention_mask = attention_mask.cuda(args.local_rank)
        priors = priors.cuda(args.local_rank)
        preds = preds.cuda(args.local_rank)
    
    with torch.no_grad():
        scores = model((input_ids, token_type_ids, attention_mask))
        scores = torch.softmax(scores, 0)
        # priors = torch.pow(priors,0.1)

        if scores.min() < 0:
            scores -= scores.min()
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

        scores = scores / torch.sum(scores)
        # scores = scores[:,1]
        index = torch.argmax(scores).item()
    # [(sense, score), ...]
    results = []
    for i, w in enumerate(querys[qname]):
        results.append((w, round(scores[i].item(), 4)))

    return querys[qname][index], results

# get F1 score
def evaluate(gold_file, res_file):
    gold_file = os.path.abspath(gold_file)
    res_file = os.path.abspath(res_file)

    os.chdir('src/tool')
    eval_res = subprocess.Popen(
        ['java', 'Scorer', gold_file, res_file],
        stdout=subprocess.PIPE, shell=False)
    os.chdir('../..')
    
    (out, err) = eval_res.communicate()
    eval_res = out.decode("utf-8")
    eval_res = eval_res.strip().split()
    index = eval_res.index('F1=') + 1
    res = eval_res[index]
    res = res.split('%')[0]
    return float(res)

TYPE2GOLD_PATH = {
    'all': 'data/wsd/evaluation/ALL/ALL.gold.key.txt',
    'semeval2007': 'data/wsd/evaluation/semeval2007/semeval2007.gold.key.txt',
    'semeval2013': 'data/wsd/evaluation/semeval2013/semeval2013.gold.key.txt',
    'semeval2015': 'data/wsd/evaluation/semeval2015/semeval2015.gold.key.txt',
    'senseval2': 'data/wsd/evaluation/senseval2/senseval2.gold.key.txt',
    'senseval3': 'data/wsd/evaluation/senseval3/senseval3.gold.key.txt'
}

def test(model, datas, dataset, args, params):
    model = model.eval()
    results = []
    for data in datas:
        ans, res = test_iterator(model, data, dataset, args, params)
        if ans != -1:
            results.append((data['id'], data['target_sense'], res, ans, data['target_word']))
    
    test_corpus = params['data']['test_corpus']
    test_gold_path = TYPE2GOLD_PATH[test_corpus]
    test_id_prefix = '' if test_corpus == 'all' else test_corpus
    with open(os.path.join(args.save_path, 'tem.res'), 'w', encoding='utf8') as fout:
        for ele in results:
            if ele[0].startswith(test_id_prefix):
                id = ele[0] if test_corpus == 'all' else ele[0].replace(f'{test_corpus}.', '')
                fout.write(id + ' ' + ele[3] + '\n')
    test_F1 = evaluate(test_gold_path, os.path.join(args.save_path, 'tem.res'))

    dev_corpus = params['data']['dev_corpus']
    dev_gold_path = TYPE2GOLD_PATH[dev_corpus]
    dev_id_prefix = '' if dev_corpus == 'all' else dev_corpus
    with open(os.path.join(args.save_path, 'tem.res'), 'w', encoding='utf8') as fout:
        for ele in results:
            if ele[0].startswith(dev_id_prefix):
                id = ele[0] if dev_corpus == 'all' else ele[0].replace(f'{dev_corpus}.', '')
                fout.write(id + ' ' + ele[3] + '\n')
    val_F1 = evaluate(dev_gold_path, os.path.join(args.save_path, 'tem.res'))

    return test_F1, val_F1, results

def test_all(results, args):
    pre = 'data/wsd/evaluation'
    # semeval2007
    res = {}
    datasets = ['semeval2007', 'senseval2', 'senseval3', 'semeval2013', 'semeval2015']
    for key in datasets:
        with open(os.path.join(args.save_path, 'tem.res'), 'w', encoding='utf8') as fout:
            for ele in results:
                if ele[0].startswith(key):
                    fout.write(ele[0].replace(key + '.', '') + ' ' + ele[3] + '\n')
        F1 = evaluate(os.path.join(pre, key, key + '.gold.key.txt'), os.path.join(args.save_path, 'tem.res'))
        res[key] = F1
    type = {'n': 'NOUN', 'v': 'VERB', 'a': 'ADJ', 'r': 'ADV'}
    for key in type:
        with open(os.path.join(args.save_path, 'tem' + key + '.res'), 'w', encoding='utf8') as fout:
            for ele in results:
                if ele[4].endswith(key):
                    fout.write(ele[0] + ' ' + ele[3] + '\n')
        F1 = evaluate(os.path.join(pre, 'ALL', 'ALL.gold.key.txt.' + type[key]), os.path.join(args.save_path, 'tem' + key + '.res'))
        res[key] = F1
    return res

def write_result(resultname,results):
    with open(resultname,'w') as fout:
        for ele in results:
            fout.write(ele[0] + '\t' + ele[1] + '\t' + str(ele[2]) + '\t' + ele[3] + '\n')
    with open(resultname + '.key','w') as fout:
        for ele in results:
            fout.write(ele[0] + ' ' + ele[3] + '\n')

def get_new_preds(model, dataset, args, params):
    model.eval()
    new_preds = dict()
    for data in dataset.data:
        ans, res = test_iterator(model, data, dataset, args, params)
        if ans != -1:
            temp_score = [result[1] for result in res]
            tot = sum(temp_score)
            temp_score = [score / tot for score in temp_score]
            new_preds[data['id']] = temp_score
    return new_preds

TOOL_SET = {
    'get_new_preds': get_new_preds, 
    'test_all': test_all,
    'test': test,
    'test_iterator': test_iterator,
    'evaluate': evaluate,
    'write_result': write_result
}