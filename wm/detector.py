# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict, Callable

import numpy as np
from scipy import special
from scipy.optimize import fminbound

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WmDetector():
    def __init__(self, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317, 
            vocab_size: int = 50304, 
        ):
        # model config
        self.vocab_size = vocab_size
        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding 
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    
    def get_seed_rng(self, input_ids: torch.LongTensor) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i.item()) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids)
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0]
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed)
        return seed

    def aggregate_scores(self, scores: List[List[np.array]], aggregation: str = 'mean') -> List[float]:
        """Aggregate scores along a text."""
        scores = np.asarray(scores)
        if aggregation == 'sum':
           return [ss.sum(axis=0) for ss in scores]
        elif aggregation == 'mean':
            return [ss.mean(axis=0) if ss.shape[0]!=0 else np.ones(shape=(self.vocab_size)) for ss in scores]
        elif aggregation == 'max':
            return [ss.max(axis=0) for ss in scores]
        else:
             raise ValueError(f'Aggregation {aggregation} not supported.')

    def get_scores_by_t(
        self, 
        neox_args, 
        tokens_id: torch.Tensor, 
        scoring_method: str="none",
        payload_max: int = 0
    ) -> List[np.array]:
        """
        Get score increment for each token in list of texts.
        Args:
            texts: list of texts
            scoring_method: 
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            payload_max: maximum number of messages 
        Output:
            score_lists: list of [np array of score increments for every token and payload] for each text
        """
        bsz = neox_args.batch_size
        total_len = neox_args.seq_length

        score_lists = []
        for ii in range(bsz):
            start_pos = self.ngram + 51 # be different based on different situation 
            rts = []
            seen_ntuples = set()
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = tokens_id[ii, cur_pos-self.ngram:cur_pos] # h
                if scoring_method == 'v1':
                    tup_for_unique = tuple(ngram_tokens)
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                elif scoring_method == 'v2':
                    tup_for_unique = tuple(ngram_tokens + tokens_id[ii][cur_pos:cur_pos+1])
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                rt = self.score_tok(ngram_tokens, tokens_id[ii][cur_pos]) 
                rt = rt.numpy()[:payload_max+1]
                rts.append(rt)
            score_lists.append(rts)
        return score_lists

    def get_pvalues(
            self, 
            scores: List[np.array], 
            eps: float=1e-200
        ) -> np.array:
        """
        Get p-value for each text.
        Args:
            score_lists: list of [list of score increments for each token] for each text
        Output:
            pvalues: np array of p-values for each text and payload
        """
        pvalues = []
        scores = np.asarray(scores) # bsz x ntoks x payload_max
        for ss in scores:
            ntoks = ss.shape[0]
            scores_by_payload = ss.sum(axis=0) if ntoks!=0 else np.zeros(shape=ss.shape[-1]) # payload_max
            pvalues_by_payload = [self.get_pvalue(score, ntoks, eps=eps) for score in scores_by_payload]
            pvalues.append(pvalues_by_payload)
        return np.asarray(pvalues) # bsz x payload_max

    def get_pvalues_by_t(self, scores: List[float], eps: float=1e-200) -> List[float]:
        """Get p-value for each text."""
        pvalues = []
        cum_score = 0
        cum_toks = 0
        for ss in scores:
            cum_score += ss
            cum_toks += 1
            pvalue = self.get_pvalue(cum_score, cum_toks, eps)
            pvalues.append(pvalue)
        return pvalues
    
    def score_tok(self, ngram_tokens: List[int], token_id: int):
        """ for each token in the text, compute the score increment """
        raise NotImplementedError
    
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ compute the p-value for a couple of score and number of tokens """
        raise NotImplementedError

class MarylandDetector(WmDetector):

    def __init__(self, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            vocab_size: int = 50304, 
            gamma: float = 0.25, 
            delta: float = 5.0, 
            **kwargs):
        super().__init__(ngram, seed, seeding, salt_key, vocab_size, **kwargs)
        self.gamma = gamma
        self.delta = delta
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = 1 if token_id in greenlist else 0 
        The last line shifts the scores by token_id. 
        ex: scores[0] = 1 if token_id in greenlist else 0
            scores[1] = 1 if token_id in (greenlist shifted of 1) else 0
            ...
        The score for each payload will be given by scores[payload]
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        scores = torch.zeros(self.vocab_size)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n toks in the greenlist
        scores[greenlist] = 1 
        return scores.roll(-token_id.item()) 
                
    def get_pvalue(self, score: int, ntoks: int, eps: float):
        """ from cdf of a binomial distribution """
        pvalue = special.betainc(score, 1 + ntoks - score, self.gamma)
        return max(pvalue, eps)

class MarylandDetectorZ(WmDetector):

    def __init__(self, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            vocab_size: int = 50304, 
            gamma: float = 0.5, 
            delta: float = 1.0, 
            **kwargs):
        super().__init__(ngram, seed, seeding, salt_key, vocab_size, **kwargs)
        self.gamma = gamma
        self.delta = delta
    
    def score_tok(self, ngram_tokens, token_id):
        """ same as MarylandDetector but using zscore """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        scores = torch.zeros(self.vocab_size)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n
        scores[greenlist] = 1
        return scores.roll(-token_id)
                
    def get_pvalue(self, score: int, ntoks: int, eps: float):
        """ from cdf of a normal distribution """
        zscore = (score - self.gamma * ntoks) / np.sqrt(self.gamma * (1 - self.gamma) * ntoks)
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)

class TransformDetector(WmDetector):

    def __init__(self, 
            n: int,
            key: int,
            T: int, 
            vocab_size: int = 0, 
            **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.key = key
        self.T = T
        self.vocab_size = vocab_size
        self._setup_watermark_key()
    
    def _setup_watermark_key(self):
        generator = torch.Generator()
        torch.manual_seed(self.key)
        seeds = torch.randint(2**32, (self.T,))
        generators = [generator.manual_seed(int(seed.item())) for seed in seeds]

        # Keys used in Generation 
        self.pis = torch.stack([torch.randperm(self.vocab_size, generator=g) for g in generators]).cuda()   # [T, vocab_size]
        self.xis = torch.stack([torch.rand((self.n, 1), generator=g) for g in generators]).cuda()           # [T, n, 1])

    def permutation_test(self, 
                        tokens,
                        batch_idx, 
                        n_runs=100,
                        max_seed=100000):
                        
        m = tokens.shape[0]
        test_result = self.phi(tokens=tokens,
                            batch_idx=batch_idx, 
                            k=m,                                          
                            normalize=True,)

        generator = torch.Generator()
        p_val = 0

        for run in range(n_runs): 
            pi = torch.randperm(self.vocab_size)

            rand_tokens = torch.argsort(pi)[tokens]

            seed = torch.randint(high=max_seed, size=(1,)).item()
            generator.manual_seed(int(seed))

            rand_pi = torch.randperm(self.vocab_size, generator=generator)
            rand_xi = torch.rand((self.n, 1), generator=generator)

            null_result = self.phi(tokens=rand_tokens,
                            batch_idx=batch_idx, 
                            k=m,  
                            normalize=True, 
                            pi=rand_pi, 
                            xi=rand_xi,)
            # assuming lower test values indicate presence of watermark
            p_val += (null_result <= test_result).float() / n_runs 
        
        return p_val

    def phi(self, 
            tokens,
            batch_idx, 
            k, 
            normalize=False, 
            pi = None, 
            xi = None):                                                        

        if pi is None and xi is None: 
            pi = self.pis[batch_idx % self.T, :]      #[vocab size]
            xi = self.xis[batch_idx % self.T, :]      #[n, 1]

        tokens = (torch.argsort(pi)[tokens])          #[m]
        if normalize:
            tokens = tokens.float() / self.vocab_size

        A = self.adjacency(tokens,xi, k)
        closest = torch.min(A,axis=1)[0]

        return torch.min(closest)

    def adjacency(self, tokens, xi, k):
        m = tokens.shape[0]

        A = torch.empty(size=(m-(k-1),self.n))
        for i in range(m-(k-1)):
            for j in range(self.n):
                A[i][j] = self.transform_score(tokens[i:i+k],xi[(j+torch.arange(k))%self.n]) # transform_edit_score for edit 

        return A # [m-(k-1), n] --> [1, n]

    def transform_score(self, tokens, xi):
        return torch.pow(torch.linalg.norm(tokens-xi.squeeze(),ord=1),1)

    def transform_edit_score(self, tokens, xi, gamma=0.4):
        xi = xi.cpu()
        tokens = tokens.cpu()
        return transform_levenshtein(tokens.numpy(),xi.squeeze().numpy(),gamma)

class OpenaiDetector(WmDetector):

    def __init__(self, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317, 
            vocab_size: int = 50304, 
            **kwargs):
        super().__init__(ngram, seed, seeding, salt_key, vocab_size, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        """ rt = -log(1 - rt[xt]]) """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng) # n
        rt = -(1 - rs).log()[token_id]
        return rt.item()
 
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ from cdf of a gamma distribution """
        pvalue = special.gammaincc(ntoks, score)
        return max(pvalue, eps)