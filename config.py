import os
import argparse
from enum import Enum, auto

# --- Enums ---
class DatasetType(Enum):
    ALPACA = auto()
    C4 = auto()   

class OptimizerType(Enum):
    NORMALIZED = auto()
    SGD = auto()
    ADAM = auto() 

class WatermarkType(Enum):
    KIRCHENBAUER = auto()
    KUDITIPUDI = auto()
    GUMBEL = auto()

class Config:
    def __init__(self):
        # --- Paths ---
        self.GPU = 0
        self.PROCESS = 0
        self.CACHE_DIR = f"./huggingface_cache_g{self.GPU}_p{self.PROCESS}"
        self.OUTPUT_DIR = f"./output_g{self.GPU}_p{self.PROCESS}"
        self.DATASET_DIR = "dataset.json"

        # --- Federated Learning Hyperparameters ---
        self.DIRTY_WORKER = 2
        self.N_WORKER = 30
        self.N_ROUNDS = 200
        self.WM_SIZE = 1024
        self.WHETHER_FILTER = False

        # --- Finetune Hyperparameters ---
        self.SEED = 42
        self.TRAINING_SIZE = 15360 
        self.TOTAL_SIZE = 16896
        self.BATCH_SIZE = 8
        self.SEQ_LEN = 1024 
        self.EPOCHS = 1
        self.LR = 1e-5
        self.WEIGHT_DECAY = 0.0     
        self.MAX_GRAD_NORM = None  

        self.WHETHER_LOAD_MODEL = False
        self.WHETHER_LOAD_WM = False
        self.WHETHER_SAVE_WM = False
        self.WHETHER_SYNTHETIC = False 
        self.CHECKPOINT_FREQ = 25

        # --- Watermark Hyperparameters ---
        self.START_POS = 20 
        self.CHUNK_SIZE = 64

        self.NGRAM = 4
        self.WM_SEED = 0
        self.SEEDING = 'hash'
        self.HASH_KEY = 35317
        # --- Watermark KIRCHENBAUER ---
        self.GAMMA = 0.25
        self.DELTA = 3.0
        self.PAYLOAD = 0
        self.TEMPERATURE = 0.8
        self.TOP_P = 0.95
        # --- Watermark KUDITIPUDI ---
        self.ROBUST_N    = self.SEQ_LEN 
        self.ROBUST_T    = self.WM_SIZE
        self.ROBUST_KEY  = 42 

        # --- Current Configuration ---
        self.DATASET = DatasetType.C4
        self.WATERMARK = WatermarkType.KIRCHENBAUER         
        self.CLIENT_OPTIMIZER = OptimizerType.NORMALIZED
        self.SERVER_OPTIMIZER = OptimizerType.ADAM
        self.MODEL_CHECKPOINT = "EleutherAI/pythia-70m-deduped"

        os.environ["PL_GLOBAL_SEED"] = str(self.SEED)
        os.environ["PL_SEED_WORKERS"] = "1"

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key.upper()) and value is not None:
                # Handle enum values specially
                if key.upper() == 'DATASET':
                    setattr(self, key.upper(), DatasetType[value.upper()])
                elif key.upper() == 'WATERMARK':
                    setattr(self, key.upper(), WatermarkType[value.upper()])
                elif key.upper() == 'CLIENT_OPTIMIZER':
                    setattr(self, key.upper(), OptimizerType[value.upper()])
                elif key.upper() == 'SERVER_OPTIMIZER':
                    setattr(self, key.upper(), OptimizerType[value.upper()])
                else:
                    setattr(self, key.upper(), value)
        
        if hasattr(args, 'dataset_path') and args.dataset_path is not None:
            self.DATASET_DIR = args.dataset_path

        if 'gpu' in vars(args) or 'process' in vars(args):
            self.CACHE_DIR = f"./huggingface_cache_g{self.GPU}_p{self.PROCESS}"
            self.OUTPUT_DIR = f"./output_g{self.GPU}_p{self.PROCESS}"

        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser(description='Training Configuration')

        parser.add_argument('--process', type=int, help='Process index')
        parser.add_argument('--whether_load_model', action='store_true', help='Whether to load model')
        parser.add_argument('--whether_load_wm', action='store_true', help='Whether to load watermark')
        parser.add_argument('--whether_save_wm', action='store_true', help='Whether to save watermark')
        parser.add_argument('--whether_synthetic', action='store_true', help='Whether all workers use synthetic data')

        parser.add_argument('--dataset', type=str, choices=[d.name.lower() for d in DatasetType],
                            help='Dataset type (c4 or alpaca)')
        parser.add_argument('--dataset_path', type=str, 
                            help='Path to dataset directory')
        parser.add_argument('--client_optimizer', type=str, choices=[o.name.lower() for o in OptimizerType],
                            help='Client optimizer type (normalized or adam)')
        parser.add_argument('--server_optimizer', type=str, choices=[o.name.lower() for o in OptimizerType],
                            help='Server optimizer type (normalized or adam)')
        parser.add_argument('--model_checkpoint', type=str, 
                            help='HuggingFace model checkpoint (e.g., "EleutherAI/pythia-160m-deduped")')
        parser.add_argument('--watermark', type=str, choices=[w.name.lower() for w in WatermarkType],
                            help='Watermark type (kirchenbauer or kuditipudi)')
        
        parser.add_argument('--n_worker', type=int, help='Number of workers in federated learning')
        parser.add_argument('--dirty_worker', type=int, help='Number of dirty workers in federated learning')
        parser.add_argument('--wm_size', type=int, help='WM size')
        parser.add_argument('--training_size', type=int, help='Training size')
        parser.add_argument('--total_size', type=int, help='Total size')
        parser.add_argument('--chunk_size', type=int, help='Chunk size')
        parser.add_argument('--whether_filter', action='store_true', help='Whether to filter')
        
        return parser

    def log_all_args(self):
        print("=== Training configuration ===")
        for key in sorted(self.__dict__):
            print(f"{key} = {self.__dict__[key]}")
        print("=============================")
        
# Create global config instance
config = Config()