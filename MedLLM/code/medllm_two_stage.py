#!/usr/local/bin/python3.12 -u
"""
MedLLM Fresh Training with PPO - Two-Stage Training Pipeline
Stage 1: Supervised Fine-tuning 
Stage 2: PPO-based Reinforcement Learning
"""

import os
import sys
import json
import torch
import argparse
import logging
import signal
import time
import shutil
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import random
import subprocess
from sklearn.metrics import accuracy_score

# Set environment variables BEFORE importing anything
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medllm_fresh_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import required packages
try:
    from unsloth import FastLanguageModel
    from datasets import Dataset, load_dataset, concatenate_datasets
    from transformers import TrainingArguments, set_seed, TrainerCallback
    from trl import SFTTrainer, PPOConfig, PPOTrainer
    import wandb
    try:
        from llama_cpp import Llama
        LLAMA_CPP_AVAILABLE = True
    except ImportError:
        logger.warning("llama-cpp-python not available, will use HF format for PPO")
        LLAMA_CPP_AVAILABLE = False
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Installing dependencies...")
    os.system("pip install -q unsloth transformers datasets trl wandb llama-cpp-python")
    from unsloth import FastLanguageModel
    from datasets import Dataset, load_dataset, concatenate_datasets
    from transformers import TrainingArguments, set_seed, TrainerCallback
    from trl import SFTTrainer, PPOConfig, PPOTrainer
    import wandb
    try:
        from llama_cpp import Llama
        LLAMA_CPP_AVAILABLE = True
    except ImportError:
        logger.warning("llama-cpp-python not available, will use HF format for PPO")
        LLAMA_CPP_AVAILABLE = False

class OptimizedConfig:
    """Optimized configuration for fresh training"""
    
    # Model configuration
    MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    MAX_SEQ_LENGTH = 2048
    
    # LoRA configuration - optimized
    LORA_R = 16
    LORA_ALPHA = 8
    LORA_DROPOUT = 0.0
    TARGET_MODULES = ["q_proj", "k_proj"]
    
    # Training configuration - optimized for stability
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 3e-6
    MAX_GRAD_NORM = 0.5
    NUM_EPOCHS = 5
    WARMUP_STEPS = 100
    SAVE_STEPS = 500  # Save less frequently to avoid checkpoint issues
    EVAL_STEPS = 500
    LOGGING_STEPS = 50
    
    # Learning rate schedule
    LR_SCHEDULER_TYPE = "cosine_with_restarts"
    
    # Memory optimization
    FP16 = True
    GRADIENT_CHECKPOINTING = "unsloth"
    OPTIM = "adamw_torch"
    
    # Dataset configuration
    MAX_SAMPLES_PER_DATASET = 50000
    DATASETS_TO_USE = ["medmcqa", "medqa"]
    
    # PPO RL Configuration
    PPO_LEARNING_RATE = 5e-7
    PPO_EPOCHS = 2
    PPO_BATCH_SIZE = 2
    PPO_SUBSET_SIZE = 500
    PPO_OUTPUT_DIR = "./medllm_ppo_output"

class MedicalDataProcessor:
    """Optimized data processor"""
    
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
    def create_prompt_template(self, question: str, answer: str) -> str:
        """Create optimized prompt template"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a knowledgeable medical AI assistant. Provide accurate, evidence-based medical information. Always recommend consulting healthcare professionals for personal medical advice.<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer}<|eot_id|>"""
    
    def validate_example(self, text: str) -> bool:
        """Validate example quality"""
        if not text or len(text.strip()) < 50:
            return False
        if "<|start_header_id|>" not in text or "<|end_header_id|>" not in text:
            return False
        if len(text) > self.max_length * 2:
            return False
        return True
    
    def process_medqa(self, max_samples: Optional[int] = None) -> Dataset:
        """Process MedQA dataset"""
        logger.info("Processing MedQA dataset...")
        try:
            dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
            if max_samples:
                dataset = dataset.select(range(min(len(dataset), max_samples)))
                
            def format_example(example):
                try:
                    question = example['question']
                    options = example['options']
                    answer_idx = example['answer_idx']
                    
                    if answer_idx not in options:
                        return None
                        
                    answer = options[answer_idx]
                    
                    formatted_q = f"{question}\n\nOptions:\n"
                    for key in ['A', 'B', 'C', 'D']:
                        if key in options:
                            formatted_q += f"{key}: {options[key]}\n"
                    
                    formatted_a = f"The correct answer is {answer_idx}: {answer}"
                    text = self.create_prompt_template(formatted_q, formatted_a)
                    
                    if not self.validate_example(text):
                        return None
                    
                    return {'text': text, 'dataset': 'medqa'}
                except Exception as e:
                    logger.warning(f"Error formatting MedQA example: {e}")
                    return None
            
            processed_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
            processed_dataset = processed_dataset.filter(lambda x: x is not None and x.get('text') is not None)
            
            logger.info(f"✅ MedQA processed: {len(processed_dataset)} valid examples")
            return processed_dataset
            
        except Exception as e:
            logger.error(f"Error processing MedQA: {e}")
            return None
    
    def process_medmcqa(self, max_samples: Optional[int] = None) -> Dataset:
        """Process MedMCQA dataset"""
        logger.info("Processing MedMCQA dataset...")
        try:
            dataset = load_dataset("openlifescienceai/medmcqa", split="train")
            if max_samples:
                dataset = dataset.select(range(min(len(dataset), max_samples)))
            
            def format_example(example):
                try:
                    question = example['question']
                    options = [example['opa'], example['opb'], example['opc'], example['opd']]
                    answer_idx = example['cop']
                    answer = options[answer_idx]
                    
                    formatted_q = f"{question}\n\nOptions:\n"
                    for i, opt in enumerate(options):
                        formatted_q += f"{chr(65+i)}: {opt}\n"
                    
                    formatted_a = f"The correct answer is {chr(65+answer_idx)}: {answer}"
                    if example.get('exp'):
                        formatted_a += f"\n\nExplanation: {example['exp']}"
                    
                    text = self.create_prompt_template(formatted_q, formatted_a)
                    
                    if not self.validate_example(text):
                        return None
                    
                    return {'text': text, 'dataset': 'medmcqa'}
                except Exception as e:
                    logger.warning(f"Error formatting MedMCQA example: {e}")
                    return None
            
            processed_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
            processed_dataset = processed_dataset.filter(lambda x: x is not None and x.get('text') is not None)
            
            logger.info(f"✅ MedMCQA processed: {len(processed_dataset)} valid examples")
            return processed_dataset
            
        except Exception as e:
            logger.error(f"Error processing MedMCQA: {e}")
            return None
    
    def prepare_datasets(self, config: OptimizedConfig) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets"""
        datasets = []
        
        if "medqa" in config.DATASETS_TO_USE:
            medqa = self.process_medqa(config.MAX_SAMPLES_PER_DATASET)
            if medqa:
                datasets.append(medqa)
        
        if "medmcqa" in config.DATASETS_TO_USE:
            medmcqa = self.process_medmcqa(config.MAX_SAMPLES_PER_DATASET)
            if medmcqa:
                datasets.append(medmcqa)
        
        if not datasets:
            raise ValueError("No datasets could be loaded!")
        
        combined_dataset = concatenate_datasets(datasets)
        split = combined_dataset.train_test_split(test_size=0.05, seed=42)
        
        logger.info(f"✅ Dataset prepared - Train: {len(split['train'])}, Val: {len(split['test'])}")
        return split['train'], split['test']

    def prepare_ppo_dataset(self, subset_size: int = 500) -> List[Dict]:
        """Prepare subset of MedQA for PPO training"""
        logger.info(f"Preparing PPO dataset with {subset_size} examples...")
        try:
            dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
            # Sample random subset
            indices = random.sample(range(len(dataset)), min(subset_size, len(dataset)))
            subset = dataset.select(indices)
            
            ppo_data = []
            for example in subset:
                question = example['question']
                options = example['options'] 
                answer_idx = example['answer_idx']
                correct_answer = options[answer_idx]
                
                # Create query prompt
                query = f"{question}\n\nOptions:\n"
                for key in ['A', 'B', 'C', 'D']:
                    if key in options:
                        query += f"{key}: {options[key]}\n"
                query += "\nWhat is the correct answer?"
                
                ppo_data.append({
                    'query': query,
                    'correct_answer': answer_idx,
                    'reference': correct_answer
                })
            
            logger.info(f"✅ PPO dataset prepared: {len(ppo_data)} examples")
            return ppo_data
            
        except Exception as e:
            logger.error(f"Error preparing PPO dataset: {e}")
            return []

def compute_reward(predictions: List[str], references: List[str]) -> List[float]:
    """Compute rewards for PPO training"""
    rewards = []
    for pred, ref in zip(predictions, references):
        # Extract answer from prediction (look for A, B, C, D)
        pred_clean = pred.strip().upper()
        ref_clean = ref.strip().upper()
        
        # Check for exact match of answer choice
        if any(choice in pred_clean for choice in ['A', 'B', 'C', 'D']):
            # Find the choice in prediction
            pred_choice = None
            for choice in ['A', 'B', 'C', 'D']:
                if choice in pred_clean:
                    pred_choice = choice
                    break
            
            if pred_choice == ref_clean:
                rewards.append(1.0)  # Exact match
            else:
                rewards.append(0.1)  # Wrong answer
        else:
            rewards.append(0.1)  # No clear answer choice
    
    return rewards

def convert_to_gguf_if_needed(model_path: str) -> str:
    """Convert model to GGUF format if needed"""
    gguf_path = str(Path(model_path) / "model.gguf")
    
    if Path(gguf_path).exists():
        logger.info(f"GGUF model already exists at {gguf_path}")
        return gguf_path
    
    try:
        logger.info(f"Converting model to GGUF format...")
        convert_script = "/home/abhishek/models/medllm_new/llama.cpp/convert_hf_to_gguf.py"
        
        if Path(convert_script).exists():
            cmd = f"python {convert_script} {model_path} --outfile {gguf_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Model converted to GGUF: {gguf_path}")
                return gguf_path
            else:
                logger.error(f"GGUF conversion failed: {result.stderr}")
                return model_path
        else:
            logger.warning(f"Convert script not found at {convert_script}")
            return model_path
            
    except Exception as e:
        logger.error(f"Error during GGUF conversion: {e}")
        return model_path

class TrainingMonitor(TrainerCallback):
    """Enhanced training monitoring"""
    
    def __init__(self):
        self.start_time = time.time()
        self.step_times = []
        
    def on_step_end(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        
        # Monitor gradients
        grad_norm = logs.get("grad_norm")
        if grad_norm is not None and isinstance(grad_norm, float):
            if grad_norm != grad_norm:  # NaN check
                logger.error("🚨 NaN gradient detected. Stopping training.")
                control.should_training_stop = True
            elif grad_norm > 100:
                logger.warning(f"⚠️ Large gradient norm: {grad_norm:.4f}")
        
        # Track speed
        current_time = time.time()
        self.step_times.append(current_time)
        if len(self.step_times) > 100:
            self.step_times = self.step_times[-100:]
        
        # Log progress
        if state.global_step % 50 == 0 and len(self.step_times) > 1:
            avg_step_time = (self.step_times[-1] - self.step_times[0]) / (len(self.step_times) - 1)
            steps_per_hour = 3600 / avg_step_time if avg_step_time > 0 else 0
            logger.info(f"📊 Step {state.global_step}: {steps_per_hour:.1f} steps/hour")
        
        # Memory monitoring
        if state.global_step % 100 == 0 and torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"🧠 GPU Memory: {memory_used:.2f}GB used, {memory_cached:.2f}GB cached")

class FreshTrainer:
    """Fresh training implementation"""
    
    def __init__(self, config: OptimizedConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        # Clear old output directory to start completely fresh
        if self.output_dir.exists():
            logger.info(f"Clearing old training directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.ppo_trainer = None
        
        # Signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        
    def _handle_signal(self, signum, frame):
        """Handle interruption signals"""
        logger.info(f"Received signal {signum}. Saving model...")
        if self.trainer:
            self.trainer.save_model(str(self.output_dir / "interrupted_save"))
        sys.exit(0)
    
    def setup_model(self):
        """Setup fresh model"""
        logger.info("🚀 Loading fresh model...")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.MODEL_NAME,
            max_seq_length=self.config.MAX_SEQ_LENGTH,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.LORA_R,
            target_modules=self.config.TARGET_MODULES,
            lora_alpha=self.config.LORA_ALPHA,
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info("✅ Fresh model loaded successfully")
        
    def train(self):
        """Start fresh training"""
        processor = MedicalDataProcessor(self.tokenizer, self.config.MAX_SEQ_LENGTH)
        train_dataset, eval_dataset = processor.prepare_datasets(self.config)
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=self.config.WARMUP_STEPS,
            num_train_epochs=self.config.NUM_EPOCHS,
            learning_rate=self.config.LEARNING_RATE,
            max_grad_norm=self.config.MAX_GRAD_NORM,
            fp16=self.config.FP16,
            bf16=False,
            logging_steps=self.config.LOGGING_STEPS,
            save_steps=self.config.SAVE_STEPS,
            eval_steps=self.config.EVAL_STEPS,
            eval_strategy="steps",
            save_strategy="steps",
            optim=self.config.OPTIM,
            weight_decay=0.01,
            lr_scheduler_type=self.config.LR_SCHEDULER_TYPE,
            seed=42,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            report_to="wandb",
            run_name=f"medllm-fresh-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.MAX_SEQ_LENGTH,
            args=training_args,
            packing=False,
        )
        
        self.trainer.add_callback(TrainingMonitor())
        
        logger.info("🏃 Starting fresh training...")
        start_time = time.time()
        
        try:
            # NO resume_from_checkpoint - completely fresh start
            self.trainer.train()
            self.trainer.save_model(str(self.output_dir / "final_model"))
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Fresh training completed in {elapsed_time/3600:.2f} hours")
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            self.trainer.save_model(str(self.output_dir / "error_save"))
            raise
    
    def setup_ppo_model(self, sft_model_path: str):
        """Setup model for PPO training"""
        logger.info("🚀 Setting up model for PPO training...")
        
        try:
            # Load the fine-tuned model
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=sft_model_path,
                max_seq_length=self.config.MAX_SEQ_LENGTH,
                dtype=torch.float16,
                load_in_4bit=True,
            )
            
            # Prepare for inference/PPO
            FastLanguageModel.for_inference(self.model)
            
            logger.info("✅ PPO model setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up PPO model: {e}")
            raise
    
    def run_rl(self, sft_model_path: str):
        """Run PPO-based reinforcement learning"""
        logger.info("🎯 Starting PPO-based RL training...")
        
        # Setup PPO model
        self.setup_ppo_model(sft_model_path)
        
        # Prepare PPO dataset
        processor = MedicalDataProcessor(self.tokenizer, self.config.MAX_SEQ_LENGTH)
        ppo_data = processor.prepare_ppo_dataset(self.config.PPO_SUBSET_SIZE)
        
        if not ppo_data:
            logger.error("No PPO data available, skipping RL training")
            return
        
        # Initialize PPO trainer
        ppo_config = PPOConfig(
            model_name="medllm-sft",
            learning_rate=self.config.PPO_LEARNING_RATE,
            ppo_epochs=self.config.PPO_EPOCHS,
            batch_size=self.config.PPO_BATCH_SIZE,
            mini_batch_size=1,
            gradient_accumulation_steps=2,
            optimize_cuda_cache=True,
            log_with="wandb",
            tracker_project_name="medllm-ppo",
            remove_unused_columns=False,
        )
        
        # Create PPO output directory
        ppo_output_dir = Path(self.config.PPO_OUTPUT_DIR)
        ppo_output_dir.mkdir(exist_ok=True)
        
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=None,  # Will use the same model as reference
            tokenizer=self.tokenizer,
        )
        
        logger.info(f"Starting PPO training with {len(ppo_data)} examples...")
        
        # PPO training loop
        batch_size = self.config.PPO_BATCH_SIZE
        for epoch in range(self.config.PPO_EPOCHS):
            logger.info(f"PPO Epoch {epoch + 1}/{self.config.PPO_EPOCHS}")
            
            # Process in batches
            for i in range(0, len(ppo_data), batch_size):
                batch = ppo_data[i:i + batch_size]
                queries = [item['query'] for item in batch]
                references = [item['reference'] for item in batch]
                
                # Generate responses
                query_tensors = [self.tokenizer.encode(q, return_tensors="pt") for q in queries]
                response_tensors = self.ppo_trainer.generate(
                    query_tensors,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode responses
                responses = [self.tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
                
                # Compute rewards
                rewards = compute_reward(responses, references)
                reward_tensors = [torch.tensor(r, dtype=torch.float) for r in rewards]
                
                # PPO step
                stats = self.ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
                
                # Log stats
                if i % 10 == 0:
                    logger.info(f"Batch {i//batch_size + 1}: Mean reward = {sum(rewards)/len(rewards):.3f}")
        
        # Save PPO model
        self.ppo_trainer.save_model(str(ppo_output_dir / "final_ppo_model"))
        logger.info(f"✅ PPO training completed, model saved to {ppo_output_dir}")

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="MedLLM Two-Stage Training: SFT + PPO")
    parser.add_argument('--output-dir', type=str, default='./medllm_fresh_output')
    parser.add_argument('--wandb-project', type=str, default='medllm-fresh')
    parser.add_argument('--skip-sft', action='store_true', help='Skip SFT and go directly to PPO')
    parser.add_argument('--sft-model-path', type=str, help='Path to SFT model for PPO (if skipping SFT)')
    parser.add_argument('--ppo-only', action='store_true', help='Run only PPO training')
    
    args = parser.parse_args()
    
    # Create trainer
    config = OptimizedConfig()
    trainer = FreshTrainer(config, args.output_dir)
    
    if args.ppo_only:
        # Run only PPO
        logger.info("🎯 Running PPO-only mode")
        sft_model_path = args.sft_model_path or str(Path(args.output_dir) / "final_model")
        
        # Initialize wandb for PPO
        wandb.init(
            project="medllm-ppo",
            config={
                "model": OptimizedConfig.MODEL_NAME,
                "hardware": "RTX 3070Ti",
                "training_type": "ppo_only",
                "ppo_learning_rate": config.PPO_LEARNING_RATE,
                "ppo_epochs": config.PPO_EPOCHS,
                "ppo_batch_size": config.PPO_BATCH_SIZE,
            }
        )
        
        trainer.run_rl(sft_model_path)
        
    elif args.skip_sft:
        # Skip SFT, run PPO on existing model
        logger.info("🎯 Skipping SFT, running PPO on existing model")
        sft_model_path = args.sft_model_path or str(Path(args.output_dir) / "final_model")
        
        # Initialize wandb for PPO
        wandb.init(
            project="medllm-ppo", 
            config={
                "model": OptimizedConfig.MODEL_NAME,
                "hardware": "RTX 3070Ti",
                "training_type": "ppo_after_sft",
                "ppo_learning_rate": config.PPO_LEARNING_RATE,
                "ppo_epochs": config.PPO_EPOCHS,
                "ppo_batch_size": config.PPO_BATCH_SIZE,
            }
        )
        
        trainer.run_rl(sft_model_path)
        
    else:
        # Run full two-stage training
        logger.info("🏥 Running two-stage training: SFT + PPO")
        
        # Stage 1: SFT
        wandb.init(
            project=args.wandb_project,
            config={
                "model": OptimizedConfig.MODEL_NAME,
                "hardware": "RTX 3070Ti", 
                "training_type": "two_stage_sft_ppo",
                "stage": "sft",
                "optimizations": "All fixes applied from start",
            }
        )
        
        trainer.setup_model()
        trainer.train()
        wandb.finish()
        
        # Stage 2: PPO
        logger.info("🎯 Starting Stage 2: PPO-based RL training...")
        
        wandb.init(
            project="medllm-ppo",
            config={
                "model": OptimizedConfig.MODEL_NAME,
                "hardware": "RTX 3070Ti",
                "training_type": "two_stage_sft_ppo", 
                "stage": "ppo",
                "ppo_learning_rate": config.PPO_LEARNING_RATE,
                "ppo_epochs": config.PPO_EPOCHS,
                "ppo_batch_size": config.PPO_BATCH_SIZE,
            }
        )
        
        sft_model_path = str(Path(args.output_dir) / "final_model")
        trainer.run_rl(sft_model_path)
    
    wandb.finish()

if __name__ == "__main__":
    logger.info("🏥 MedLLM Two-Stage Training Pipeline: SFT + PPO")
    main()
