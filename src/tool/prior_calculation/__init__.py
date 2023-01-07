from typing import Literal, List, Union, IO, Set
import math


class BaseCalculator(object):
    
    def __init__(self, model_path: str, corpus_path: str):
        self.model_path = model_path
        self.corpus_path = corpus_path

        self.model = None
        self.corpus = None

    def fit(self, sents: List[str]):
        raise NotImplementedError()

    def calculate_score(self, querys: List[str], candidates: List[str], whitening: bool = False) -> float:
        raise NotImplementedError()

    def _load_stopwords(self, stopwords_path: str):
        with open(stopwords_path, 'r', encoding='utf8') as fi:
            self.stopwords = [line.strip() for line in fi.readlines()]

class BaseFomalizer(object):

    def __init__(self, corpus_paths: Union[List[str], str]):
        assert type(corpus_paths) in [list, str]

        if isinstance(corpus_paths, list):
            self.corpus_paths = corpus_paths
        elif isinstance(corpus_paths, str):
            self.corpus_paths = [corpus_paths]

        self.corpus = []

        for corpus_path in self.corpus_paths:
            self.load_corpus(corpus_path)

        self.process_corpus()

    def load_corpus(self, corpus_path: str = None):
        raise NotImplementedError()

    def process_corpus(self):
        raise NotImplementedError()

    def _dump(self, f: IO[str], text: str, cache: Set[str]):
        if text not in cache:
            cache.add(text)
            f.write('{}\n'.format(text))

    def dump_corpus(self, dump_path: str):
        raise NotImplementedError()

    def dump_querys_and_candidates(self, dump_path: str):
        raise NotImplementedError()
