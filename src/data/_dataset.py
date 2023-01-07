import re
import random
import torch.distributed as dist
from transformers import BertTokenizer
from src.tool import DEFAULT_CONFIG

REMOVE_PATTERN = re.compile(r'[\t\n\r\f`]')

class _Dataset(object):
    def __init__(self, params):
        self.params = params
        self._build_dataset()
        self._init_generate_part()

    def _init_generate_part(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            self.params['model'].get('bert_path', DEFAULT_CONFIG['model']['bert_path']))

        self.batch_size = self.params['data'].get(
                            'mini_batch', DEFAULT_CONFIG['data']['mini_batch'])
        self.select = list(range(len(self.data)))
        self.index = 0
        random.shuffle(self.select)
        self.index_generater = self._index_generator()

    def __len__(self):
        return len(self.data)

    def _index_generator(self):
        assert dist.is_available()
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        self.index = rank
        while True:
            if self.index >= len(self.data):
                self.index = rank
                random.shuffle(self.select)
            start = self.index
            end = start + world_size * self.batch_size
            self.index = end
            yield self.select[start:end:world_size]
    
    def generate_bertinput(self, query, candidate):
        raise NotImplementedError()

    def generate_batch(self):
        raise NotImplementedError()
    
    def _build_dataset(self):
        raise NotImplementedError()

    @staticmethod
    def process_sentence(sentence):
        return REMOVE_PATTERN.sub(' ', sentence.strip())