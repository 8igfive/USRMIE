import torch
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from typing import List, cast
from sentence_transformers import util, SentenceTransformer
from transformers import BertTokenizer, BertForNextSentencePrediction, BertLMHeadModel, AutoTokenizer, AutoModel

from . import BaseCalculator

import pdb

DEBUG = False

class USECalculator(BaseCalculator):
    def __init__(self, model_path: str, corpus_path: str = None):
        super().__init__(model_path, corpus_path)
        self.model = hub.load(model_path)

        # whitening part
        self.xtx_sum = cast(np.ndarray, None)
        self.x_sum = cast(np.ndarray, None)
        self.sample_count = 0

        if DEBUG:
            self.embeds = cast(np.ndarray, None)

        self.bias = cast(np.ndarray, None)
        self.W = cast(np.ndarray, None)

    def fit(self, sents: List[str]):
        # reinit
        self.bias = cast(np.ndarray, None)
        self.W = cast(np.ndarray, None)

        sents_embd = self.model(sents).numpy()
        
        if self.sample_count == 0:
            f_dim = sents_embd.shape[-1]
            self.xtx_sum = np.zeros(shape=(f_dim, f_dim))
            self.x_sum = np.zeros(shape=(f_dim,))

        if DEBUG:
            if self.sample_count == 0:
                self.embeds = sents_embd
            else:
                self.embeds = np.concatenate([self.embeds, sents_embd], axis=0)
        
        self.sample_count += len(sents)
        self.xtx_sum += np.matmul(sents_embd.T, sents_embd)
        self.x_sum += sents_embd.sum(axis=0)
        
    def calculate_score(self, querys: List[str], candidates: List[str], whitening: bool = False) -> float:
        querys_emb = self.model(querys).numpy()
        cands_emb = self.model(candidates).numpy()

        if whitening:
            if self.bias is None:
                mu = self.x_sum / self.sample_count
                self.bias = -mu
                cov = self.xtx_sum / self.sample_count - np.matmul(mu[:, None], mu[None, :])
                U, S, _ = np.linalg.svd(cov)
                self.W = np.matmul(U, np.diag(1 / np.sqrt(S)))

                if DEBUG:
                    new_embeds = np.matmul(self.embeds + self.bias, self.W)
                    mean = new_embeds.mean(axis=0)
                    cov = np.matmul((new_embeds - mean).T, new_embeds - mean) / new_embeds.shape[0]
                    print(mean)
                    print(cov)
                    pdb.set_trace() # FIXME
                
            querys_emb = np.matmul((querys_emb + self.bias), self.W)
            cands_emb = np.matmul((cands_emb + self.bias), self.W)

        return util.cos_sim(querys_emb, cands_emb)

class SBertCalculator(BaseCalculator):
    def __init__(self, model_path: str, corpus_path: str = None):
        super().__init__(model_path, corpus_path)
        self.model = SentenceTransformer(self.model_path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        # whitening part
        self.xtx_sum = cast(np.ndarray, None)
        self.x_sum = cast(np.ndarray, None)
        self.sample_count = 0

        if DEBUG:
            self.embeds = cast(np.ndarray, None)

        self.bias = cast(np.ndarray, None)
        self.W = cast(np.ndarray, None)

    def fit(self, sents: List[str]):

        self.bias = cast(np.ndarray, None)
        self.W = cast(np.ndarray, None)

        with torch.no_grad():
            sents_embd = self.model.encode(sents)

        if self.sample_count == 0:
            f_dim = sents_embd.shape[-1]
            self.xtx_sum = np.zeros(shape=(f_dim, f_dim))
            self.x_sum = np.zeros(shape=(f_dim,))

        if DEBUG:
            if self.sample_count == 0:
                self.embeds = sents_embd
            else:
                self.embeds = np.concatenate([self.embeds, sents_embd], axis=0)

        self.sample_count += len(sents)
        self.xtx_sum += np.matmul(sents_embd.T, sents_embd)
        self.x_sum += sents_embd.sum(axis=0)


    def calculate_score(self, querys: List[str], candidates: List[str], whitening: bool = False) -> float:
        with torch.no_grad():
            querys_emb = self.model.encode(querys)
            cands_emb = self.model.encode(candidates)

        if whitening:
            if self.bias is None:
                mu = self.x_sum / self.sample_count
                self.bias = -mu
                cov = self.xtx_sum / self.sample_count - np.matmul(mu[:, None], mu[None, :])
                U, S, _ = np.linalg.svd(cov)
                self.W = np.matmul(U, np.diag(1 / np.sqrt(S)))

                if DEBUG:
                    new_embeds = np.matmul(self.embeds + self.bias, self.W)
                    mean = new_embeds.mean(axis=0)
                    cov = np.matmul((new_embeds - mean).T, new_embeds - mean) / new_embeds.shape[0]
                    print(mean)
                    print(cov)
                    pdb.set_trace() # FIXME

            querys_emb = np.matmul(querys_emb + self.bias, self.W)
            cands_emb = np.matmul(cands_emb + self.bias, self.W)
                
        return util.cos_sim(querys_emb, cands_emb)

class BertNSPCalculator(BaseCalculator):
    def __init__(self, model_path: str, corpus_path: str = None):
        super().__init__(model_path, corpus_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForNextSentencePrediction.from_pretrained(model_path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    def calculate_score(self, querys: List[str], candidates: List[str], whitening: bool = False) -> float:
        with torch.no_grad():
            inputs = []
            for query in querys:
                for cand in candidates:
                    inputs.append((query, cand))
            inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt') 
            if torch.cuda.is_available():
                inputs = {key: ipt.cuda() for key, ipt in inputs.items()}
            logits = self.model(**inputs)['logits'][:, 0].view(len(querys), len(candidates))
            return logits

class SimCSECalculator(BaseCalculator):
    def __init__(self, model_path: str, corpus_path: str = None):
        super().__init__(model_path, corpus_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        # whitening part
        self.xtx_sum = cast(torch.Tensor, None)
        self.x_sum = cast(torch.Tensor, None)
        self.sample_count = 0

        if DEBUG:
            self.embeds = cast(torch.Tensor, None)

        self.bias = cast(torch.Tensor, None)
        self.W = cast(torch.Tensor, None)

    def fit(self, sents: List[str]):

        self.bias = cast(torch.Tensor, None)
        self.W = cast(torch.Tensor, None)

        with torch.no_grad():
            sents_input = self.tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
            if torch.cuda.is_available():
                sents_input = {key: qi.cuda() for key, qi in sents_input.items()}
            sents_embd = self.model(**sents_input, output_hidden_states=True, return_dict=True).pooler_output

        if self.sample_count == 0:
            f_dim = sents_embd.shape[-1]
            self.xtx_sum = torch.zeros(f_dim, f_dim).to(sents_embd.device)
            self.x_sum = torch.zeros(f_dim).to(sents_embd.device)
        
        if DEBUG:
            if self.sample_count == 0:
                self.embeds = sents_embd
            else:
                self.embeds = torch.cat([self.embeds, sents_embd], dim=0)

        self.sample_count += len(sents)
        self.xtx_sum += sents_embd.T.matmul(sents_embd)
        self.x_sum += sents_embd.sum(dim=0)

    def calculate_score(self, querys: List[str], candidates: List[str], whitening: bool = False) -> float:
        with torch.no_grad():
            querys_input = self.tokenizer(querys, padding=True, truncation=True, return_tensors='pt')
            cands_input = self.tokenizer(candidates, padding=True, truncation=True, return_tensors='pt')
            if torch.cuda.is_available():
                querys_input = {key: qi.cuda() for key, qi in querys_input.items()}
                cands_input = {key: ci.cuda() for key, ci in cands_input.items()}

            querys_emb = self.model(**querys_input, output_hidden_states=True, return_dict=True).pooler_output
            cands_emb = self.model(**cands_input, output_hidden_states=True, return_dict=True).pooler_output

        if whitening:
            if self.bias is None:
                mu = self.x_sum / self.sample_count
                self.bias = -mu
                cov = self.xtx_sum / self.sample_count - mu[:, None].matmul(mu[None, :])
                U, S, _ = torch.svd(cov)
                self.W = U.matmul(torch.diag(1 / torch.sqrt(S)))

                if DEBUG:
                    new_embeds = torch.matmul(self.embeds + self.bias, self.W)
                    mean = new_embeds.mean(dim=0)
                    cov = torch.matmul((new_embeds - mean).T, new_embeds - mean) / new_embeds.shape[0]
                    print(mean)
                    print(cov)
                    pdb.set_trace() # FIXME
                
            querys_emb = (querys_emb + self.bias).matmul(self.W)
            cands_emb = (cands_emb + self.bias).matmul(self.W)

        return util.cos_sim(querys_emb, cands_emb)