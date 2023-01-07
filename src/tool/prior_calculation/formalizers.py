import json
import os
import re
from tqdm import tqdm
from typing import List, Union, cast
from . import BaseFomalizer

REMOVE_PATTERN = re.compile(r'[\t\n\r\f`]')
ABBREVIATIONS = {' no.', ' st.', ' mr.', ' ms.', 'u.s.', ' ca.'}
SEPS = {'.', '?', '!'}

class QAFormalizer(BaseFomalizer):
    
    def __init__(self, corpus_paths: Union[List[str], str]):
        self.punctuations = (',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '{', '}')
        super().__init__(corpus_paths=corpus_paths)
    
    def load_corpus(self, corpus_path: str = None):
        with open(corpus_path, 'r', encoding='utf8') as fi:
            self.corpus += json.load(fi)

    def process_corpus(self):
        for sample in tqdm(self.corpus):
            sample['query'] = REMOVE_PATTERN.sub(' ', sample['query'])
            for candidate in sample['candidates']:
                candidate['subject'] = REMOVE_PATTERN.sub(' ', candidate['subject'])
                candidate['body'] = REMOVE_PATTERN.sub(' ', candidate['body'])

    def dump_corpus(self, dump_path: str):
        if not dump_path:
            dump_path = f'{os.path.splitext(self.corpus_paths[-1])[0]}_dump_corpus'
        alread_dump = set()
        with open(dump_path, 'w', encoding='utf8') as fo:
            for sample in self.corpus:
                self._dump(fo, sample['query'], alread_dump)
                for candidate in sample['candidates']:
                    if candidate['subject']:
                        self._dump(fo, candidate['subject'], alread_dump)
                    if candidate['body']:
                        self._dump(fo, candidate['body'], alread_dump)
                        
    def dump_querys_and_candidates(self, dump_path: str):
        if not dump_path:
            dump_path = f'{os.path.splitext(self.corpus_paths[-1])[0]}_qac.json'
        json_res = []
        for sample in self.corpus:
            formalized_sample = {}
            formalized_sample['id'] = sample['id']
            formalized_sample['query'] = sample['query']
            formalized_sample['candidates'] = []
            for candidate in sample['candidates']:
                formalized_candidate = {}
                formalized_candidate['cid'] = candidate['cid']
                formalized_candidate['label'] = candidate['label']
                temp_content = candidate['subject']
                if temp_content:
                    if temp_content.endswith(self.punctuations):
                        temp_content += ' '
                    else:
                        temp_content += '. '
                temp_content += candidate['body']
                formalized_candidate['content'] = temp_content
                formalized_sample['candidates'].append(formalized_candidate)
            json_res.append(formalized_sample)
        with open(dump_path, 'w', encoding='utf8') as fo:
            json.dump(json_res, fo, ensure_ascii=False, indent=4)

class PRFormalizer(BaseFomalizer):
    
    def load_corpus(self, corpus_path: str = None):
        with open(corpus_path, 'r', encoding='utf8') as fi:
            self.corpus += json.load(fi)

    def process_corpus(self):
        for sample in self.corpus:
            sample['query'] = REMOVE_PATTERN.sub(' ', sample['query'])
            for candidate in sample['candidates']:
                candidate['body'] = REMOVE_PATTERN.sub(' ', candidate['body'])

    @staticmethod
    def passage_split(passage: str) -> List[str]:
        REMOVE_PATTERN.sub(' ', passage)

        sentences = []

        left = 0
        # single_quote_flag = 0
        double_quote_flag = 0
        for right in range(len(passage)):

            if passage[right] == '"':
                double_quote_flag = 1 - double_quote_flag

            if right == len(passage) - 1 or \
                (passage[right] in SEPS and \
                (passage[right + 1] == ' ' or passage[right + 1] == '"')):

                sentence = passage[left: right + 1].strip()

                if len(sentence) >= 10:
                    '''
                    for i in range(3, len(sentence) + 1): # 考虑 . 前面有一个 ' ' 的情况
                        if sentence[-i] == '.' or sentence[-i] == ' ':
                            break
                    if i <= 3: # ABC.
                        continue
                    '''
                    if right != len(passage) - 1 and \
                        (sentence[-4: ].lower() in ABBREVIATIONS or 
                        (double_quote_flag and passage[right + 1] != '"') or 
                         sentence[-3] == ' ' or sentence[-3] == '.'):
                            continue
                    left = right + 1
                    sentences.append(sentence)
        
        return sentences

    def dump_corpus(self, dump_path: str):
        if not dump_path:
            dump_path = f'{os.path.splitext(self.corpus_paths[-1])[0]}_dump_corpus'
        alread_dump = set()
        with open(dump_path, 'w', encoding='utf8') as fo:
            for sample in self.corpus:
                self._dump(fo, sample['query'], alread_dump)
                for candidate in sample['candidates']:
                    passage = cast(str, candidate['body'])
                    sentences = self.passage_split(passage)
                    for sentence in sentences:
                        self._dump(fo, sentence, alread_dump)
                        
    def dump_querys_and_candidates(self, dump_path: str):
        if not dump_path:
            dump_path = f'{os.path.splitext(self.corpus_paths[-1])[0]}_qac.json'

        json_res = [
            {
                'id': sample['id'],
                'query': sample['query'],
                'candidates': [
                    {
                        'cid': candidate['cid'],
                        'label': candidate['label'],
                        'content': candidate['body']
                    }
                    for candidate in sample['candidates']
                ]
            } for sample in self.corpus
        ]

        with open(dump_path, 'w', encoding='utf8') as fo:
            json.dump(json_res, fo, ensure_ascii=False, indent=4)

class WSDFormalizer(BaseFomalizer):
    def __init__(self, corpus_paths: Union[List[str], str]):
        # corpus_paths should be arranged as [query_path, gloss_path, test_path]
        self.punctuations = {',', '.', ':', ';', '?', '!'}
        super().__init__(corpus_paths=corpus_paths)

    def load_corpus(self, corpus_path: str = None):
        '''
        Args:
            corpus_path:  0 for query_path, 1 for gloss_path, 2 for test_path
        '''
        if 'query' in corpus_path:
            with open(corpus_path, 'r', encoding='utf8') as fi:
                self.querys = json.load(fi)
        elif 'gloss' in corpus_path:
            with open(corpus_path, 'r', encoding='utf8') as fi:
                self.gloss = json.load(fi)
        else:
            with open(corpus_path, 'r', encoding='utf8') as fi:
                self.test_data = json.load(fi)               # [{}, ...]
        
            self.corpus = []
            for data in self.test_data:
                sample = {}
                sample['id'] = data['id']
                temp_query = ' '.join(data['context'])
                sample['query'] = ''.join([ch for seq, ch in enumerate(temp_query) 
                                    if seq == len(temp_query) - 1 or \
                                        ch != ' ' or temp_query[seq + 1] not in self.punctuations])
                sample['candidates'] = []
                target_word = data['target_word']
                word = target_word.split('#')[0]
                target_sense = data['target_sense']
                senses = self.querys[target_word]
                for sense in senses:
                    candidate = {}
                    candidate['cid'] = sense
                    candidate['label'] = '1' if sense == target_sense else '0'
                    candidate['subject'] = word
                    candidate['body'] = '{}.'.format(' '.join(self.gloss[sense]))
                    sample['candidates'].append(candidate)
                self.corpus.append(sample)
            
    def process_corpus(self):
        for sample in self.corpus:
            sample['query'] = REMOVE_PATTERN.sub(' ', sample['query'])
            for candidate in sample['candidates']:
                candidate['body'] = REMOVE_PATTERN.sub(' ', candidate['body'])

    def dump_corpus(self, dump_path: str):
        if not dump_path:
            dump_path = f'{os.path.splitext(self.corpus_paths[-1])[0]}_dump_corpus'
        already_dump = set()
        with open(dump_path, 'w', encoding='utf8') as fo:
            for sample in self.corpus:
                self._dump(fo, sample['query'], already_dump)
                for candidate in sample['candidates']:
                    content = '{}: {}'.format(candidate['subject'], candidate['body'])
                    self._dump(fo, content, already_dump)
    
    def dump_querys_and_candidates(self, dump_path: str):
        if not dump_path:
            dump_path = f'{os.path.splitext(self.corpus_paths[-1])[0]}_qac.json'

        json_res = [
            {
                'id': sample['id'],
                'query': sample['query'],
                'candidates': [
                    {
                        'cid': candidate['cid'],
                        'label': candidate['label'],
                        'content': '{}: {}'.format(candidate['subject'], candidate['body'])
                    } for candidate in sample['candidates']
                ]
            } for sample in self.corpus
        ]

        with open(dump_path, 'w', encoding='utf8') as fo:
            json.dump(json_res, fo, ensure_ascii=False, indent=4)
  