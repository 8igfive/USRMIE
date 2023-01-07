import copy
import math
import os
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from transformers import BertTokenizer, BertModel

logger = logging.getLogger(__name__)

class Model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.bert = BertModel.from_pretrained(params['bert_path'])
        # fine-tuning the bert parameters
        unfreeze_layers = ['pooler'] + ['layer.'+str(11 - i) for i in range(params['layers'])]
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

        self.classifier = nn.Linear(768,1)
        # nn.init.xavier_uniform_(self.classifier.weight,gain=params['alpha'])
        parameters = torch.load(os.path.join(params['bert_path'], 'pytorch_model.bin'))
        needed_parameters = {
            'weight': parameters['cls.seq_relationship.weight'][:1, :],
            'bias': parameters['cls.seq_relationship.bias'][:1]
        }
        self.classifier.load_state_dict(needed_parameters)

    def forward(self, inputs):
        # unpack inputs
        tokens, token_ids,masks = inputs

        # ipnut embs
        output = self.bert(tokens, token_type_ids=token_ids, attention_mask=masks,return_dict=True)

        text_cls = output['pooler_output']

        score = self.classifier(text_cls).squeeze(-1)
        return score 

