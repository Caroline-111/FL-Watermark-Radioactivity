import os
import re
import gc
import torch
import nltk
import evaluate
import pandas as pd
from itertools import islice
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from torch.optim import Optimizer
import shutil
from config import config, DatasetType, OptimizerType, WatermarkType

import numpy as np
import random
from transformers import set_seed
from nltk.tokenize import sent_tokenize
from wm import (WmGenerator, MarylandGenerator, MarylandDetector, MarylandDetectorZ,
                TransformGenerator, TransformDetector, OpenaiGenerator, OpenaiDetector)
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import json 
import statistics
from pathlib import Path

nltk.download("punkt")
nltk.download('punkt_tab') 
rouge_score = evaluate.load("rouge")

# --- Deterministic ---
def set_seeds_and_determinism(seed):
    # --- Environment ---
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)

    # --- Python/Random ---
    random.seed(seed)
    
    # --- NumPy ---
    np.random.seed(seed)
    
    # --- PyTorch ---
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    # --- CuDNN ---
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    # --- Hugging Face ---
    set_seed(seed)

# --- Dataset Utilities ---
def load_c4_dataset(tokenizer, config):
    full_dataset = load_dataset("allenai/c4", "realnewslike", streaming=True, split="train")
    shuffled_dataset = full_dataset.shuffle(seed=config.SEED, buffer_size=100_000)
    subset = list(islice(shuffled_dataset, config.TOTAL_SIZE))
    
    def tokenize_c4(examples):
        tokenized = tokenizer(
            examples["text"],
            max_length=config.SEQ_LEN,
            truncation=True,
            padding="max_length",
        )
        tokenized["token_count"] = [sum(mask) for mask in tokenized["attention_mask"]]
        return tokenized

    subset_dataset = Dataset.from_list(subset)
    tokenized_dataset = subset_dataset.map(tokenize_c4, 
                                            batched=True, 
                                            remove_columns=["text", "timestamp", "url"])
    split_dataset = tokenized_dataset.train_test_split(train_size=config.TRAINING_SIZE, seed=config.SEED) 
    return split_dataset

def load_alpaca_dataset(tokenizer, config):
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    ds = dataset.shuffle(seed=config.SEED).select(range(config.TOTAL_SIZE))
    
    def tokenize(x):
        context_part = f"\nInput:\n{x['input']}\n" if x['input'] else ""
        
        # Text until Prompt: 
        prefix_text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
        ### Instruction:\n{x['instruction']}
        {context_part}
        ### Response:\n"""
        
        # Full text
        full_text = prefix_text + f"""{x['output']}
        ### End
        """
        tokenized = tokenizer(full_text, 
                            max_length=config.SEQ_LEN, 
                            truncation=True, 
                            padding="max_length",)
                            
        tokenized["token_count"] = sum(tokenized["attention_mask"])
        prefix_tokenized = tokenizer(prefix_text)
        prompt_size = len(prefix_tokenized["input_ids"])
        tokenized["prompt_size"] = prompt_size
        
        return tokenized

    tokenized_dataset = ds.map(tokenize, 
                                remove_columns=["instruction", "input", "output", "text"], 
                                load_from_cache_file=True,  # Ensures same tokenization order
                                desc="Running tokenizer",)
    tokenized_dataset = tokenized_dataset.map(lambda x, idx: {"original_idx": idx}, with_indices=True)
    split_dataset = tokenized_dataset.train_test_split(train_size=config.TRAINING_SIZE, seed=config.SEED)
    return split_dataset

def get_dataset(tokenizer, config):
    if config.DATASET == DatasetType.ALPACA:
        return load_alpaca_dataset(tokenizer, config)
    elif config.DATASET == DatasetType.C4:
        return load_c4_dataset(tokenizer, config)

# --- Model Utilities ---
def load_model(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_CHECKPOINT,
        cache_dir=config.CACHE_DIR,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token 
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_CHECKPOINT,
        cache_dir=config.CACHE_DIR,
        torch_dtype=torch.float32,
    )

    return model, tokenizer

class NormalizedSGD(Optimizer):
    def __init__(self, params, lr=None):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if (grad_norm := torch.norm(grad, p=2.0)) > 0:
                    grad.div_(grad_norm)
                p.data.add_(grad, alpha=-group['lr'])
        return loss

def get_optimizer(model, lr, optimizer_t):
    return NormalizedSGD(model.parameters(), lr) if optimizer_t == OptimizerType.NORMALIZED else None

# --- Watermark ---
def clean_workers_generate(batch, WM_Model, config):
    device = WM_Model.device
    input_ids = torch.tensor(batch["input_ids"]).to(device)
    attention_mask = torch.tensor(batch["attention_mask"]).to(device)
    bsz, seq_len = attention_mask.shape
    
    if config.DATASET == DatasetType.ALPACA: 
        prompt_size = torch.tensor(batch["prompt_size"], dtype=torch.long).to(device) + config.NGRAM
        token_count = torch.tensor(batch["token_count"], dtype=torch.long).to(device)
        max_gen_len = 506
        end_token_reserve = 6 
        # define start_pos and total_len
        start_pos = min(prompt_size)
        token_count = torch.minimum(prompt_size + max_gen_len, 
                           torch.tensor(seq_len - end_token_reserve).to(device))
        total_len = max(token_count)

        # update attention_mask
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1) 
        prompt_size_ex = prompt_size.unsqueeze(1)
        token_count_ex = token_count.unsqueeze(1) 

        attention_mask = (positions >= prompt_size_ex) & (positions < token_count_ex)
        attention_mask = attention_mask.long() 

    elif config.DATASET == DatasetType.C4:
        start_pos = config.START_POS
        total_len = max(batch["token_count"])
    
    past_key_values = None
    prev_pos = 0
    for cur_pos in range(start_pos, total_len):
        with torch.no_grad():
            outputs = WM_Model(input_ids[:, prev_pos:cur_pos], 
                                        past_key_values=past_key_values,
                                        use_cache=True,)     
            past_key_values = outputs.past_key_values  

            if config.WATERMARK == WatermarkType.KIRCHENBAUER or config.WATERMARK == WatermarkType.GUMBEL: 
                if config.TEMPERATURE > 0: 
                    logits = outputs.logits[:, -1, :] / config.TEMPERATURE
                    probs = torch.softmax(logits, dim=-1)
                    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                    probs_cumsum = torch.cumsum(probs_sort, dim=-1)
                    mask = (probs_cumsum - probs_sort) < config.TOP_P
                    probs_sort[~mask] = 0.0
                    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
                    if config.WATERMARK == WatermarkType.KIRCHENBAUER:
                        next_token_idx = torch.multinomial(probs_sort, 1)
                        next_tokens = torch.gather(probs_idx, -1, next_token_idx).squeeze()
                    elif config.WATERMARK == WatermarkType.GUMBEL:
                        next_token_idx = torch.argmax(probs_sort, dim=-1, keepdim=True)
                        next_tokens = torch.gather(probs_idx, -1, next_token_idx).squeeze() 
                else: 
                    next_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
            elif config.WATERMARK == WatermarkType.KUDITIPUDI:
                probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze() 

            # Update sequence in-place
            update_mask = (attention_mask[:, cur_pos] == 1)
            input_ids[update_mask, cur_pos] = next_tokens[update_mask]
            prev_pos = cur_pos
    
    if config.DATASET == DatasetType.ALPACA: 
        # If EOS appears, we should stop early 
        eos_mask = (input_ids == 0)
        first_eos_positions = torch.argmax(eos_mask.int(), dim=1)
        eos_found = eos_mask.any(dim=1)
        token_count_up = torch.where(eos_found, first_eos_positions, token_count)

        end_tokens = torch.tensor([187, 50270, 4118, 8072, 187, 50270] * bsz, device=device).flatten()

        # Create masks for each segment
        token_count_ex = token_count_up.unsqueeze(1)
        mask_end = (positions >= token_count_ex) & (positions < token_count_ex + end_token_reserve)
        mask_total = positions < token_count_ex + end_token_reserve

        # Update input_ids, attention_mask, token_count
        input_ids[mask_end]  = end_tokens
        input_ids[~mask_total] = 0
        batch["input_ids"] = input_ids
        attention_mask[mask_total] = 1
        attention_mask[~mask_total] = 0
        batch["attention_mask"] = attention_mask
        batch["token_count"] = token_count_up + end_token_reserve
        
    elif config.DATASET == DatasetType.C4:
        batch["input_ids"] = input_ids
    return batch
    
def watermark_generate(train_dataset, config): 
    # Create WM Model
    WM_Model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-2.8b-deduped", 
                                                    torch_dtype=torch.float32,
                                                    ).cuda()
    WM_Model.eval()

    # Create Generator
    if config.WATERMARK == WatermarkType.KIRCHENBAUER:
        generator = MarylandGenerator(WM_Model, 
                                        config.NGRAM, 
                                        config.WM_SEED, 
                                        config.SEEDING, 
                                        config.HASH_KEY,
                                        payload=config.PAYLOAD, 
                                        gamma=config.GAMMA, 
                                        delta=config.DELTA
                                    )  
    elif config.WATERMARK == WatermarkType.KUDITIPUDI:
        generator = TransformGenerator(WM_Model, 
                                        config.NGRAM, 
                                        config.WM_SEED, 
                                        config.SEEDING, 
                                        config.HASH_KEY,
                                        n=config.ROBUST_N, 
                                        key=config.ROBUST_KEY, 
                                        T=config.ROBUST_T, 
                                    )
    elif config.WATERMARK == WatermarkType.GUMBEL:
        generator = OpenaiGenerator(WM_Model, 
                                        config.NGRAM, 
                                        config.WM_SEED, 
                                        config.SEEDING, 
                                        config.HASH_KEY,
                                    )

    # Watermark Dataset Generation
    first_n = train_dataset.select(range(config.WM_SIZE))     
    wm_dataset_chunks = []
    for i in range(0, len(first_n), config.CHUNK_SIZE):                         
        print(f"WM Worker - Generate Batch {i}", flush=True)
        chunk = first_n.select(range(i, min(i + config.CHUNK_SIZE, len(first_n))))
        
        # Generate watermarked chunk
        wm_chunk = generator.generate(
            chunk,
            config,
            chunk_num=i,                                                     
        )
        
        wm_dataset_chunks.append(wm_chunk)
    
    # Clean Dataset Generation
    if config.WM_SIZE < len(train_dataset):
        if config.WHETHER_SYNTHETIC:
            print("Clean Worker - Synthetic Data")
            generate_fn = partial(clean_workers_generate, WM_Model=WM_Model, config=config)
            raw_data = train_dataset.select(range(config.WM_SIZE, len(train_dataset)))
            remaining = raw_data.map(generate_fn, batched=True, batch_size=64)
        else: 
            print("Clean Worker - Clean Data")
            remaining = train_dataset.select(range(config.WM_SIZE, len(train_dataset)))
    else:
        remaining = None 
        print("No Clean Worker - WM_SIZE Covers Entire Dataset")

    # Combine all chunks
    if config.WM_SIZE > 0:
        wm_first_n = concatenate_datasets(wm_dataset_chunks)
        if config.WM_SIZE < len(train_dataset):
            wm_train_dataset = concatenate_datasets([wm_first_n, remaining])
        else: 
            wm_train_dataset = wm_first_n
    else:
        wm_train_dataset = remaining
        
    return wm_train_dataset

def watermark_detect(wm_train_dataset, model, config): 
    model.eval()
    # -- Create Detector ---
    if config.WATERMARK == WatermarkType.KIRCHENBAUER:
        dic = {}
        dic_r = {}
        detector = MarylandDetector(config.NGRAM, 
                                    config.WM_SEED, 
                                    config.SEEDING, 
                                    config.HASH_KEY, 
                                    vocab_size=model.config.vocab_size, 
                                    gamma=config.GAMMA) 
    elif config.WATERMARK == WatermarkType.KUDITIPUDI:
        dic =[]
        detector = TransformDetector(n=config.ROBUST_N, 
                                    key=config.ROBUST_KEY, 
                                    T=config.ROBUST_T, 
                                    vocab_size=model.config.vocab_size)
    elif config.WATERMARK == WatermarkType.GUMBEL:
        dic = {}
        dic_r = {}
        detector = OpenaiDetector(config.NGRAM, 
                                    config.WM_SEED, 
                                    config.SEEDING, 
                                    config.HASH_KEY, 
                                    vocab_size=model.config.vocab_size) 

    wm_train_dataset = list(wm_train_dataset)
    for i in range(0, len(wm_train_dataset), config.CHUNK_SIZE):
        batch = wm_train_dataset[i:i + config.CHUNK_SIZE]
        
        token_count = torch.tensor([x["token_count"] for x in batch])
        total_len = max(token_count)

        if config.DATASET == DatasetType.ALPACA: 
            start_pos = min(t['prompt_size'] for t in batch) + detector.ngram

            input_ids = pad_sequence(
                [torch.tensor(x["input_ids"]) for x in batch],
                batch_first=True,
                padding_value=0
            )
            attention_mask = pad_sequence(
                [torch.tensor(x["attention_mask"]) for x in batch],
                batch_first=True,
                padding_value=0
            )
            prompt_size = torch.tensor([x["prompt_size"] + detector.ngram for x in batch])
        elif config.DATASET == DatasetType.C4:
            start_pos = config.START_POS
    
            input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
            attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])

        # Compute logits
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        predictions = logits.argmax(dim=-1)

        if config.WATERMARK == WatermarkType.KIRCHENBAUER or config.WATERMARK == WatermarkType.GUMBEL:
            current_batch = {bs:set() for bs in range(config.CHUNK_SIZE)}

            k = config.NGRAM
            for j in range(start_pos-k-1, total_len-k-1):
                slices = torch.cat((input_ids[:, j+1:j+k+1], predictions[:, j+k:j+k+1]), dim=-1).tolist() # [CHUNK_SIZE, k+1]
                for num_slice, slice in enumerate(slices): #bsz of [k+1]
                    # print(f"detect batch: {num_slice}")
                    if config.DATASET == DatasetType.ALPACA: 
                        startpoint = prompt_size[num_slice]-k-1
                        endpoint = token_count[num_slice]-k-7
                    elif config.DATASET == DatasetType.C4:
                        startpoint = 0
                        endpoint = token_count[num_slice]-k-1

                    if j >= startpoint and j < endpoint:
                        tup = torch.tensor(slice)

                        if not str(tup[:-1]) in current_batch[num_slice]: # de-duplication
                            current_batch[num_slice].add(str(tup[:-1])) 

                            if not str(tup) in dic.keys():
                                seed = detector.get_seed_rng(tup[:-1]) 
                                # print(f"detect seq idx: {j+k}, seed: {seed}")
                                # print(f"expect output: {tup[-1]}, ngrams: {tup[:-1]}")
                                detector.rng.manual_seed(seed)

                                if config.WATERMARK == WatermarkType.KIRCHENBAUER: 
                                    vocab_permutation = torch.randperm(detector.vocab_size, generator=detector.rng)
                                    greenlist = vocab_permutation[:int(detector.gamma * detector.vocab_size)] # gamma * n are in greenlist
                                    # print(f"detect greenlist: {greenlist.shape}, {greenlist}")
                                    rt = 1 if tup[-1] in greenlist else 0
                                    r = rt
                                elif config.WATERMARK == WatermarkType.GUMBEL: 
                                    rs = torch.rand(detector.vocab_size, generator=detector.rng) # n
                                    rt = -(1 - rs).log()[tup[-1]]
                                    r = rt.item()

                                dic[str(tup)] = r
                                if config.WATERMARK == WatermarkType.KIRCHENBAUER: 
                                    dic_r[str(tup)] = r
                                elif config.WATERMARK == WatermarkType.GUMBEL: 
                                    dic_r[str(tup)] = rs[tup[-1]].item()
        elif config.WATERMARK == WatermarkType.KUDITIPUDI:
            for j in range(len(batch)): 
                if config.DATASET == DatasetType.ALPACA: 
                    startpoint = prompt_size[j]
                elif config.DATASET == DatasetType.C4:
                    startpoint = config.START_POS
                p_val = detector.permutation_test(predictions[j, startpoint-1:token_count[j]-1], i+j)
                dic.append(p_val)

        del input_ids, attention_mask, logits, predictions
        torch.cuda.empty_cache()
        gc.collect()
    
    if config.WATERMARK == WatermarkType.KIRCHENBAUER:
        mean_r = np.mean(list(dic_r.values()))
        pvalues = detector.get_pvalues_by_t(dic.values())
        print(f"KIRCHENBAUER Detect mean_r: {mean_r}; p_val: {pvalues[-1]}")
    elif config.WATERMARK == WatermarkType.KUDITIPUDI:
        median_r = statistics.median(dic)
        print(f"KUDITIPUDI Detect median p_val: {median_r:.6f}")
    elif config.WATERMARK == WatermarkType.GUMBEL:
        mean_r = np.mean(list(dic_r.values()))
        pvalues = detector.get_pvalues_by_t(dic.values())
        print(f"GUMBEL Detect mean_r: {mean_r}; p_val: {pvalues[-1]}")
    model.train()

# --- Save Checkpoints ---
def save_round_checkpoint(global_model, round_num, config):
    round_model_dir = os.path.join(config.OUTPUT_DIR, f"round_{round_num}")
    os.makedirs(round_model_dir, exist_ok=True)
    global_model.save_pretrained(round_model_dir)

def save_filtering_metrics(round_num, recall, precision, filename="filtering_metrics.jsonl"):
    round_data = {
        "round": round_num,
        "recall": recall,
        "precision": precision,
    }
    
    # Append to JSONL file
    with open(filename, 'a') as f:
        f.write(json.dumps(round_data) + '\n')