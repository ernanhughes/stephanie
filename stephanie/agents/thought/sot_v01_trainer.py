# stephanie/agents/thought/sot_v01_trainer.py
from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from .sot_v01_collator import SoTDataCollator
from .sot_v01_dataset import SoTV01Dataset
from .sot_v01_multitask import SoTMultiTaskWrapper


class SoTV01Trainer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device=="cuda" else torch.float32
        )

        # LoRA
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            task_type=TaskType.CAUSAL_LM
        )
        base = get_peft_model(base, lora_cfg)
        base.print_trainable_parameters()

        hidden = base.base_model.model.layers[-1].mlp.gate_proj.out_features if hasattr(base, "base_model") else base.config.hidden_size
        self.model = SoTMultiTaskWrapper(base, hidden_size=hidden, move_loss_weight=0.2).to(self.device)

    def train(self, train_path: str, output_dir: str, epochs: int = 2, batch_size: int = 4, lr: float = 2e-4):
        ds = SoTV01Dataset(train_path, self.tokenizer)
        collator = SoTDataCollator(self.tokenizer)

        # Custom Trainer to pass extra fields
        class MTTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                move_labels = inputs.pop("move_labels")
                prompt_lengths = inputs.pop("prompt_lengths")
                outputs = model(**inputs, labels=labels, move_labels=move_labels, prompt_lengths=prompt_lengths)
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            save_strategy="epoch",
            logging_steps=20,
            report_to="none",
            bf16=self.device=="cuda",
            fp16=False,  # bf16 preferred if available
            gradient_checkpointing=True
        )

        trainer = MTTrainer(
            model=self.model,
            args=args,
            train_dataset=ds,
            data_collator=collator,
            tokenizer=self.tokenizer
        )
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
