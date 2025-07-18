from model import PolicyModel
from dataset import get_summarize_tldr_dataloaders
from utils import setup_logging, get_device
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import os

hyperparameters = {
    "model_name": "gpt2",
    "tokenizer_name": "gpt2",
    "max_prompt_length": 512,
    "max_summary_length": 128,
    "batch_size": 2,
    "num_workers": 2,
    "learning_rate": 2e-5,
    "num_epochs": 1,
    "model_name": "Qwen/Qwen3-0.6B-Base",
    "tokenizer_name": "Qwen/Qwen3-0.6B-Base",
}

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(hyperparameters["tokenizer_name"])
    logger = setup_logging()
    logger.info("Hyperparameters: %s", hyperparameters)
    logger.info("Device: %s", get_device())

    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = get_summarize_tldr_dataloaders(
        tokenizer=tokenizer,
        max_prompt_length=hyperparameters["max_prompt_length"], 
        max_summary_length=hyperparameters["max_summary_length"],
        batch_size=hyperparameters["batch_size"], 
        num_workers=hyperparameters["num_workers"])
    logger.info("# Training Data: %d", len(train_loader.dataset))
    logger.info("# Validation Data: %d", len(val_loader.dataset))
    logger.info("# Test Data: %d", len(test_loader.dataset))

    logger.info("Loading model...")
    model = PolicyModel(
        model_name=hyperparameters["model_name"], 
        tokenizer=tokenizer, 
        max_prompt_length=hyperparameters["max_prompt_length"], 
        max_summary_length=hyperparameters["max_summary_length"],
        logger=logger)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    # best_val_loss = float("inf")
    # best_model_path = None
    for epoch in range(hyperparameters["num_epochs"]):
        train_loss = train_sft(model, train_loader, optimizer, scheduler, logger)
        # val_loss = evaluate(model, val_loader)

def train_sft(model, train_loader, optimizer, scheduler, logger):
    model.train()
    total_loss = 0
    batch_num = 0
    batch_size = train_loader.batch_size
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        batch_num = batch_num + 1
        input_ids = batch["input_ids"].to(model.model.device)
        attention_mask = batch["attention_mask"].to(model.model.device)
        label_ids = batch["label_ids"].to(model.model.device)    
        # logger.info("Input IDs: %s", input_ids.shape)
        # logger.info("Attention Mask: %s", attention_mask.shape)
        # logger.info("Labels: %s", labels.shape)
        loss = model(input_ids, attention_mask, label_ids)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        pbar.set_postfix({"Average loss": total_loss / (batch_num * batch_size)})
    logger.info("Training loss: %f", total_loss / (batch_num * batch_size))
    model_path = f"models/sft/{hyperparameters['model_name']}"
    os.makedirs(model_path, exist_ok=True)
    model.save_model(model_path)
    logger.info("Model saved to %s", model_path)

    return total_loss / (batch_num * batch_size)

if __name__ == "__main__":
    main()