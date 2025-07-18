import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel

class PolicyModel(nn.Module):
    def __init__(self, model_name, tokenizer, max_prompt_length, max_summary_length, logger):
        super().__init__()
        logger.info("Initializing base model... %s", model_name)
        base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        logger.info("freezing base model...")
        for param in base_model.parameters():
            param.requires_grad = False
        logger.info("Base model frozen")

        self.tokenizer = tokenizer

        logger.info("Initializing PEFT model...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
        )
    
        logger.info("Initializing PEFT model...")
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        logger.info("PEFT model initialized")

    def save_model(self, path):
        self.model.save_pretrained(path)
    
    def load_model(self, path):
        self.model = PeftModel.from_pretrained(self.model, path)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def generate(self, input_ids, attention_mask):
        return self.model.generate(input_ids, attention_mask=attention_mask)