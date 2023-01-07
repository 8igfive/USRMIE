from . import wsd_tool_set
from . import qa_tool_set

TOOL_SETS = {
    'wsd': wsd_tool_set.TOOL_SET,
    'qa': qa_tool_set.TOOL_SET,
}

DEFAULT_CONFIG = {
    'data': {
        'dataset': '',                      
        # batch
        'mini_batch': 1,                    
        'cand_use_method': 'normal',        
        # sample
        'neg_method': 'mix',                
        'pos_cand_num': 2,                  
        'k': 10,                            
        'neg_cand_num': 2,                  
        'pos_main_num': 2,
        'neg_main_num': None,
        'pos_main_prob': 0.75,              
        'neg_main_prob': 0.95,                      
        # data
        'modify_pred': 'nop',
        'data_type': 'testtraindev',
        'test_path': '',                    
        'dev_path': '',
        'train_path': '',
        'pred_path': '',
        # wsd exclusive
        'query_path': '',
        'gloss_path': '',
        'train_corpus': ['all'],            # all, semeval2007, semeval2013, semeval2015, senseval2, senseval3
        'test_corpus': 'all',
        'dev_corpus': 'semeval2007', 
        # tokenize
        'max_length': 512
    }, 
    'model': {
        'model_type': 'base_model',
        'dim_embed': 300,
        'hidden_dim': 300,
        'bert_path': 'bert-base-uncased',
        'layers': 6,
        'alpha': 0.003
    },
    'train': {
        'name': '',
        'optimizer': {
            'type': 'adam',
            'lr': 1e-6,
            'weight_decay': 0
        }, 
        'epoch': 100,
        'loss': {
            'type': 'nce',               # nce or infonce
            'temperature': 0.3           # temperature for InfoNCE
        }, 
        'metric': 'MAP',                 # QA: MAP, MRR, P5; WSD: F1 
        'preds_update_base': 'priors',     # '' or 'priors' 'preds'
        'preds_update_score_power': 0.1, 
        'preds_update_base_power': 1.0, 
        'sample_using_preds': False, 
        'sample_uniform': False, 
        'with_priors': False,
        'wo_estep': False, 
        'wo_source': False
    }
}