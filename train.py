import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import copy
from collections import defaultdict
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import TrainingArguments, Trainer, TrainerCallback
from config import Config, config, DatasetType, OptimizerType
from utils import *
from robust_aggregator import *

def main():
    # ===== Update configs =====
    parser = Config.get_parser()
    args = parser.parse_args()
    config.update_from_args(args)
    config.log_all_args()

    # ===== Deterministic =====
    set_seeds_and_determinism(config.SEED)

    # ===== Setup model & dataset =====
    model, tokenizer = load_model(config)

    if config.WHETHER_LOAD_MODEL: 
        final_model_path = os.path.join("./output", "final_model")
        model = AutoModelForCausalLM.from_pretrained(final_model_path)

    split_dataset = get_dataset(tokenizer, config)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
    
    # Training config
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE, 
        per_device_eval_batch_size=config.BATCH_SIZE, 
        eval_strategy="no",
        save_strategy="no",
        logging_dir="./logs",
        logging_steps=32,
        learning_rate=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        fp16=False,
        max_grad_norm=config.MAX_GRAD_NORM,
        load_best_model_at_end=False,
        report_to="none", 
        optim="sgd",
        lr_scheduler_type="constant", 

        seed=config.SEED,  # Hugging Face internal seeding
        dataloader_num_workers=0,  # Required for reproducibility
        dataloader_drop_last=True, 
        dataloader_pin_memory=False, 
        ddp_find_unused_parameters=False,  
        gradient_accumulation_steps=1,
        remove_unused_columns=True,  
        torch_compile=False,  # Disable compilation
    )

    full_train_dataset = split_dataset["train"].shuffle(seed=config.SEED) 
    full_eval_dataset = split_dataset["test"].shuffle(seed=config.SEED)

    # ---- Clean workers with synthetic data ----
    if config.WHETHER_SYNTHETIC: 
        print("Evaluation - Synthetic Data")
        if config.WHETHER_LOAD_WM: 
            full_eval_dataset = load_dataset('json', data_files='dataset_eval.json', split='train')
        else: 
            # This only works for C4!
            WM_Model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-2.8b-deduped", 
                                                    torch_dtype=torch.float32,
                                                    ).cuda()
            generate_fn = partial(clean_workers_generate, WM_Model=WM_Model, config=config)
            full_eval_dataset = full_eval_dataset.map(generate_fn, batched=True, batch_size=64)
            full_eval_list = [item for item in full_eval_dataset]  
            with open('dataset_eval.json', 'w') as f:
                json.dump(full_eval_list, f) 
            del WM_Model, full_eval_list

    # ---- P Value Generation ----
    if config.WHETHER_LOAD_WM: 
        wm_train_dataset = load_dataset('json', data_files=config.DATASET_DIR, split='train') 
        print("Load Dataset!")
    else: 
        wm_train_dataset = watermark_generate(full_train_dataset, config=config, )
        print("Generate Dataset!")
    print(wm_train_dataset)

    if config.WHETHER_SAVE_WM:
        wm_train_list = [item for item in wm_train_dataset]  
        with open(config.DATASET_DIR, 'w') as f:
            json.dump(wm_train_list, f) 
        print("Save Dataset!")

    # ---- P Value Detection ---- 
    watermark_detect(wm_train_dataset.select(range(config.WM_SIZE)), 
                model, 
                config, )

    # ---- Divide the dataset for Federated Learning ----
    worker_datasets = []
    chunk_size = len(wm_train_dataset) // config.N_WORKER
    
    for i in range(config.N_WORKER):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < config.N_WORKER - 1 else len(wm_train_dataset)
        worker_dataset = wm_train_dataset.select(range(start_idx, end_idx)) 
        worker_datasets.append(worker_dataset)
    
    # ===== Compute Initial Evaluation Loss (Pre-Trained Model) =====
    print("\n=== Evaluating Initial Pre-Trained Model ===")
    initial_trainer = Trainer(
        model=model,
        args=training_args,  # Using the same args is fine for evaluation
        train_dataset=wm_train_dataset, 
        eval_dataset=full_eval_dataset,  
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Evaluate on evaluation set
    initial_eval_results = initial_trainer.evaluate(
        metric_key_prefix="eval"
    )

    # Evaluate on training set (with prefix)
    initial_train_results = initial_trainer.evaluate(
        eval_dataset=wm_train_dataset, 
        metric_key_prefix="train"
    )

    print(f"Initial training loss (before fine-tuning): {initial_train_results['train_loss']:.4f}")
    print(f"Initial evaluation loss (before fine-tuning): {initial_eval_results['eval_loss']:.4f}")

    # ==================== Federated Learning =========================
    # Initialize global model
    global_model = copy.deepcopy(model).cuda()

    # Initialize server optimizer if using Adam
    server_optimizer = None
    if config.SERVER_OPTIMIZER == OptimizerType.ADAM:
        server_optimizer = torch.optim.AdamW(global_model.parameters(), lr=config.LR)
    
    # Early stopping variables
    best_eval_loss = float('inf')
    eval_loss_history = []
    patience = 3  # Number of rounds to wait before early stopping
    patience_counter = 0   

    for round_num in range(config.N_ROUNDS):
        print(f"\n=== Starting Federated Round {round_num + 1} ===")
        
        worker_updates = defaultdict(list)
        
        # Train each worker on their subset
        for worker_id in range(config.N_WORKER):
            print(f"Training worker {worker_id + 1}/{config.N_WORKER}")
            
            # Create worker model
            worker_model = copy.deepcopy(global_model).cuda().train()

            # Create and run trainer
            if config.CLIENT_OPTIMIZER == OptimizerType.NORMALIZED:
                trainer = Trainer(
                    model=worker_model,
                    args=training_args,
                    train_dataset=worker_datasets[worker_id],
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    optimizers=(get_optimizer(worker_model, config.LR, config.CLIENT_OPTIMIZER), None),
                )
            elif config.CLIENT_OPTIMIZER == OptimizerType.SGD:
                # Use plain SGD 
                trainer = Trainer(
                    model=worker_model,
                    args=training_args,
                    train_dataset=worker_datasets[worker_id],
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                )
            else: 
                raise ValueError(f"Unsupported Client Optimizer Type: {config.CLIENT_OPTIMIZER}")
            trainer.train()

            # Calculate parameter updates (difference between final and initial)
            with torch.no_grad():
                updates = {}
                for (name, global_param), worker_param in zip(
                    global_model.named_parameters(),
                    worker_model.parameters()
                ):
                    update = (worker_param.detach().to(torch.float64) - 
                        global_param.detach().to(torch.float64))
                    updates[name] = update.cpu() 

                # Store updates for aggregation
                for name, update in updates.items():
                    worker_updates[name].append(update)
      
            del worker_model
            torch.cuda.empty_cache()

        # Strong Byzantine Robust Aggregator 
        if config.WHETHER_FILTER: 
            filter_grad, recall, precision = robust_aggregator(worker_updates, config)
            avg_recall = sum(recall.values()) / len(recall)
            avg_precision = sum(precision.values()) / len(precision)
            print(f"avg_recall: {avg_recall}, avg_precision: {avg_precision}")
            save_filtering_metrics(round_num, recall, precision, f"filtering_metrics_g{config.GPU}_p{config.PROCESS}.jsonl")

        # Aggregate updates 
        print("Aggregating updates...")
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                if name in worker_updates:
                    # Average all worker updates for this parameter
                    if config.WHETHER_FILTER: 
                        avg_update = filter_grad[name].reshape(param.shape)
                    else: 
                        avg_update = torch.stack(worker_updates[name]).mean(0) 
                    
                    # Apply server optimizer if using Adam, otherwise direct update (FedAvg)
                    if config.SERVER_OPTIMIZER == OptimizerType.ADAM:
                        # Set gradient for server optimizer
                        param.grad = -avg_update.to(param.device).to(torch.float32)
                    else:
                        # Direct parameter update (standard FedAvg)
                        param.add_(avg_update.to(param.device).to(torch.float32))
        
        # Apply server optimizer step if using Adam
        if config.SERVER_OPTIMIZER == OptimizerType.ADAM:
            server_optimizer.step()
            server_optimizer.zero_grad()

        # Evaluate current global model
        print(f"Evaluating global model after round {round_num + 1}...")
        eval_trainer = Trainer(
            model=global_model,
            args=training_args,
            eval_dataset=full_eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        current_eval_results = eval_trainer.evaluate(metric_key_prefix="eval")
        current_eval_loss = current_eval_results['eval_loss']
        eval_loss_history.append(current_eval_loss)
        
        print(f"Round {round_num + 1} evaluation loss: {current_eval_loss:.4f}")
        
        # Early stopping logic
        if current_eval_loss < best_eval_loss:
            best_eval_loss = current_eval_loss
            patience_counter = 0
            print(f"New best evaluation loss: {best_eval_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Evaluation loss increased. Patience counter: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {round_num + 1} rounds!")
                print(f"Best evaluation loss was: {best_eval_loss:.4f}")
                break
        
        del eval_trainer
        torch.cuda.empty_cache()

        if round_num % config.CHECKPOINT_FREQ == 0:
            save_round_checkpoint(            
            global_model=global_model,
            round_num=round_num,
            config=config, 
        )

    # ================= Compute Final Evaluation Loss ================== 
    print("\n=== Final Evaluation ===")
    final_trainer = Trainer(
        model=global_model,
        args=training_args,
        train_dataset=wm_train_dataset, 
        eval_dataset=full_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Evaluate on evaluation set
    eval_results = final_trainer.evaluate(
        metric_key_prefix="eval"
    )

    # Evaluate on training set (with prefix)
    train_results = final_trainer.evaluate(
        eval_dataset=wm_train_dataset, 
        metric_key_prefix="train"
    )

    print(f"Final train loss: {train_results['train_loss']:.4f}")
    print(f"Final evaluation loss: {eval_results['eval_loss']:.4f}")
    
    # --- Save final model ---
    final_model_path = os.path.join(config.OUTPUT_DIR, "final_model")
    global_model.save_pretrained(final_model_path)

    # --- Clean up memory ---
    del model, initial_trainer, trainer, final_trainer, worker_datasets 
    gc.collect()
    torch.cuda.empty_cache()

    # ---- P Value Detection ---- 
    watermark_detect(wm_train_dataset.select(range(config.WM_SIZE)), 
                global_model, 
                config, )

if __name__ == "__main__":
    main()