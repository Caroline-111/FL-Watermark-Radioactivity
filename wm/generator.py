# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict
from datasets import Dataset
from config import DatasetType, OptimizerType, WatermarkType

import torch
from transformers import AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WmGenerator():
    def __init__(self, 
            model: AutoModelForCausalLM, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            payload: int = 0,
        ):
        # model config
        self.model = model
        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding 
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.payload = payload

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    
    def get_seed_rng(
        self, 
        input_ids: torch.LongTensor
    ) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i.item()) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids).item()
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0].item()
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed).item()
        return seed

    @torch.no_grad()
    def generate(
        self,
        train_dataset, 
        config, 
        chunk_num, 
    ) -> torch.LongTensor:
        """
        Generate text from prompts. 
        Adapted from https://github.com/facebookresearch/llama/
        """
        prev_pos = 0
        bsz = len(train_dataset)
        total_len = max(t['token_count'] for t in train_dataset)

        if config.DATASET == DatasetType.ALPACA:
            start_pos = min(t['prompt_size'] for t in train_dataset) + self.ngram 
        elif config.DATASET == DatasetType.C4:
            start_pos = config.START_POS

        wm_ids = torch.zeros((bsz, config.SEQ_LEN), device='cuda', dtype=torch.long)
        attention_mask = torch.zeros((bsz, config.SEQ_LEN), device='cuda', dtype=torch.long)

        train_data_list = [item for item in train_dataset]

        if config.DATASET == DatasetType.ALPACA:
            for i, t in enumerate(train_data_list):
                prompt_size = t['prompt_size'] + self.ngram 
                max_gen_len = 506
                end_token_reserve = 6    # Token Counts for "\n###END"

                total_size = min(prompt_size + max_gen_len, config.SEQ_LEN - end_token_reserve)

                if total_len < total_size:
                    total_len = total_size

                wm_ids[i, :prompt_size] = torch.as_tensor(t['input_ids'][:prompt_size], device='cuda')
                attention_mask[i, prompt_size:total_size] = 1
        elif config.DATASET == DatasetType.C4: 
            for i, t in enumerate(train_data_list):
                wm_ids[i] = torch.as_tensor(t['input_ids'], device='cuda')
                attention_mask[i] = torch.as_tensor(t['attention_mask'], device='cuda')

        past_key_values = None  
        for cur_pos in range(start_pos, total_len):
            # print(f"generate seq idx: {cur_pos}")                               
            outputs = self.model(wm_ids[:, prev_pos:cur_pos], 
                                        past_key_values=past_key_values,
                                        use_cache=True,)     
            past_key_values = outputs.past_key_values  

            ngram_tokens = wm_ids[:, cur_pos-self.ngram:cur_pos]
            if config.WATERMARK == WatermarkType.KIRCHENBAUER: 
                next_toks = self.sample_next(
                    outputs.logits[:, -1, :],
                    ngram_tokens,
                    config.TEMPERATURE,
                    config.TOP_P
                )
            elif config.WATERMARK == WatermarkType.KUDITIPUDI:
                next_toks = self.sample_next(
                    outputs.logits[:, -1, :],
                    ngram_tokens,
                    config.TEMPERATURE,
                    config.TOP_P, 
                    chunk_num, 
                    cur_pos, 
                )
            elif config.WATERMARK == WatermarkType.GUMBEL: 
                next_toks = self.sample_next(
                    outputs.logits[:, -1, :],
                    ngram_tokens,
                    config.TEMPERATURE,
                    config.TOP_P
                )
            
            # Update sequence in-place
            update_mask = (attention_mask[:, cur_pos] == 1)
            wm_ids[update_mask, cur_pos] = next_toks[update_mask]
            prev_pos = cur_pos

        # Update original dataset in-place
        for b in range(bsz):
            if config.DATASET == DatasetType.ALPACA:
                prompt_size = train_data_list[b]['prompt_size'] + self.ngram 
                max_gen_len = 506
                end_token_reserve = 6    # Token Counts for "\n###END"

                total_size = min(prompt_size + max_gen_len, config.SEQ_LEN - end_token_reserve)
                
                # If EOS appears, we should stop early 
                eos_positions = (wm_ids[b] == 0).nonzero()
                if len(eos_positions) > 0:
                    first_eos = eos_positions[0].item()
                    total_size = min(total_size, first_eos)

                # Update the dataset 
                end_tokens = [187, 50270, 4118, 8072, 187, 50270]
                padding_length = config.SEQ_LEN - total_size - end_token_reserve
                
                train_data_list[b]['input_ids'][:total_size] = wm_ids[b][:total_size].cpu().tolist()
                train_data_list[b]['input_ids'][total_size:total_size+end_token_reserve] = end_tokens
                train_data_list[b]['input_ids'][total_size+end_token_reserve:]      = [0] * padding_length
                train_data_list[b]['attention_mask'][:total_size+end_token_reserve] = [1] * (total_size + end_token_reserve)
                train_data_list[b]['attention_mask'][total_size+end_token_reserve:] = [0] * padding_length
                train_data_list[b]['token_count'] = total_size + end_token_reserve
            elif config.DATASET == DatasetType.C4:
                total_size = train_data_list[b]['token_count']
                train_data_list[b]['input_ids'][:total_size] = wm_ids[b][:total_size].cpu().tolist()

        train_dataset = Dataset.from_list(train_data_list)

        return train_dataset

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ) -> torch.LongTensor:
        """ Vanilla sampling with temperature and top p."""
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1) # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

class MarylandGenerator(WmGenerator):
    """ Generate text using LLaMA and Maryland's watemrarking method. """
    def __init__(self, 
            *args, 
            gamma: float = 0.5,
            delta: float = 1.0,
            **kwargs
        ):
        super().__init__(*args, **kwargs)        
        self.gamma = gamma
        self.delta = delta

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and get a seed
        - use the seed to partition the vocabulary into greenlist (gamma*V words) and blacklist 
        - add delta to greenlist words' logits
        payload (the message) is encoded by shifting the secret vector r by `payload`.
        """
        logits = self.logits_processor(logits, ngram_tokens)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1) # one hot of next token, ordered by original probs
            # print(f"next_token: {next_token}")
            next_token = torch.gather(probs_idx, -1, next_token) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

    def logits_processor(self, logits, ngram_tokens):
        """Process logits to mask out words in greenlist."""
        bsz, vocab_size = logits.shape
        logits = logits.clone()
        for ii in range(ngram_tokens.shape[0]): # batch of texts
            # print(f"generate batch: {ii}")
            seed = self.get_seed_rng(ngram_tokens[ii])
            # print(f"seed: {seed}, ngram_tokens: {ngram_tokens[ii]}")
            self.rng.manual_seed(seed)
            vocab_permutation = torch.randperm(vocab_size, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * vocab_size)] # gamma * n
            # print(f"generate greenlist: {greenlist}")
            bias = torch.zeros(vocab_size).to(logits.device) # n
            bias[greenlist] = self.delta
            bias = bias.roll(-self.payload)
            logits[ii] += bias # add bias to greenlist words
        return logits

class TransformGenerator(WmGenerator):
    """Generate watermarked text using Transform sampling method from the robust watermarking paper."""
    
    def __init__(self, 
            *args,
            n: int,                          
            key: int,
            T: int,                                                      
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.n = n
        self.key = key
        self.T = T
        self.vocab_size = self.model.config.vocab_size
        self._setup_watermark_key()
    
    def _setup_watermark_key(self):
        generator = torch.Generator()
        torch.manual_seed(self.key)
        seeds = torch.randint(2**32, (self.T,))                                 
        generators = [generator.manual_seed(int(seed.item())) for seed in seeds]

        # Generate Keys 
        self.pis = torch.stack([torch.randperm(self.vocab_size, generator=g) for g in generators]).cuda()   # [T, vocab_size]
        self.xis = torch.stack([torch.rand((self.n, 1), generator=g) for g in generators]).cuda()           # [T, n, 1])

    def sample_next(
        self,
        logits: torch.FloatTensor,
        ngram_tokens: torch.LongTensor,
        temperature: float = 0.8,
        top_p: float = 0.95,
        chunk_num: int = 0, 
        cur_pos: int = 0, 
    ) -> torch.LongTensor:
        """Sample next token using transform sampling."""
        # Hyper-parameters 
        bsz = logits.shape[0]

        probs = torch.softmax(logits, dim=-1).cuda()

        indices = torch.arange(chunk_num, chunk_num + bsz, device=self.pis.device) % self.T
        pi = self.pis[indices, :]                                              
        xi = self.xis[indices, cur_pos % self.n]

        next_tokens = self.transform_sampling(probs, 
                                                pi, 
                                                xi)

        return next_tokens.squeeze() 
    
    def transform_sampling(self, probs, pi, xi):
        cdf = torch.cumsum(torch.gather(probs, 1, pi), 1)
        xi = torch.clamp(xi, max=cdf[:, -1:])     
        return torch.gather(pi, 1, torch.searchsorted(cdf, xi))

class OpenaiGenerator(WmGenerator): 
    """
    Generate text using LLaMA and Aaronson's watermarking method.
    From ngram tokens, select the next token based on the following:
    - hash the ngram tokens and get a seed
    - use the seed to generate V random number r between [0,1]
    - select argmax ( r^(1/p) )
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ):
        bsz, vocab_size = logits.shape
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            for ii in range(ngram_tokens.shape[0]): # batch of texts
                # seed with hash of ngram tokens
                seed = self.get_seed_rng(ngram_tokens[ii])
                self.rng.manual_seed(seed)
                # generate rs randomly between [0,1]
                rs = torch.rand(vocab_size, generator=self.rng) # n
                rs = torch.Tensor(rs).to(probs_sort.device)
                rs = rs[probs_idx[ii]] 
                # compute r^(1/p)
                probs_sort[ii] = torch.pow(rs, 1/probs_sort[ii])
            # select argmax ( r^(1/p) )
            next_token = torch.argmax(probs_sort, dim=-1, keepdim=True)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token
