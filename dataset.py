import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi, get_session
from huggingface_hub.utils import hf_raise_for_status

def _patched_get_paths_info(self, repo_id, paths, *, expand=False, revision=None, repo_type=None, token=None):
    repo_type = repo_type or "model"
    revision = quote(revision, safe="") if revision is not None else "main"
    headers  = self._build_hf_headers(token=token)
    payload  = {"paths": paths if isinstance(paths, list) else [paths],
                "expand": expand}
    r = get_session().post(
        f"{self.endpoint}/api/{repo_type}s/{repo_id}/paths-info/{revision}",
        json=payload,       # ← **JSON, not form data**
        headers=headers,
    )
    hf_raise_for_status(r)
    from huggingface_hub.hf_api import RepoFile, RepoFolder
    return [RepoFile(**p) if p["type"] == "file" else RepoFolder(**p) for p in r.json()]


class SummarizeTldrDataset(Dataset):

    def __init__(self, hf_dataset, tokenizer, max_prompt_length, max_summary_length):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_summary_length = max_summary_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = "TLDR: " + item["prompt"] + " "
        label = item["label"]
        # Concatenate prompt and completion for training
        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels_ids = self.tokenizer(
            label,
            truncation=True,
            max_length=self.max_summary_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = torch.cat([prompt_ids["input_ids"], labels_ids["input_ids"]], dim=1).squeeze(0)
        attention_mask = torch.cat([prompt_ids["attention_mask"], labels_ids["attention_mask"]], dim=1).squeeze(0)
        labels_ids = input_ids.clone()
        labels_ids[:prompt_ids["input_ids"].size(1)] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": labels_ids,
            "prompt_text": prompt,
            "label_text": label,
            "input_text": prompt + label,
        }

def get_summarize_tldr_datasets(tokenizer, max_prompt_length, max_summary_length):
    """
    Loads the CarperAI/openai_summarize_tldr dataset and returns train, val, test datasets.
    """
    HfApi.get_paths_info = _patched_get_paths_info   # hot-patch

    raw_datasets = load_dataset("CarperAI/openai_summarize_tldr")
    # GPT-2 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token    
    train_dataset = SummarizeTldrDataset(raw_datasets["train"], tokenizer, max_prompt_length, max_summary_length)
    val_dataset = SummarizeTldrDataset(raw_datasets["valid"], tokenizer, max_prompt_length, max_summary_length)
    test_dataset = SummarizeTldrDataset(raw_datasets["test"], tokenizer, max_prompt_length, max_summary_length)
    return train_dataset, val_dataset, test_dataset

def get_summarize_tldr_dataloaders(tokenizer, max_prompt_length, max_summary_length, batch_size, num_workers):
    train_dataset, val_dataset, test_dataset = get_summarize_tldr_datasets(
        tokenizer=tokenizer, 
        max_prompt_length=max_prompt_length, 
        max_summary_length=max_summary_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    return train_loader, val_loader, test_loader

# returns {
#     "input_ids": <tensor>,
#     "attention_mask": <tensor>,
#     "labels": <tensor>,
#     "prompt": <str>,
#     "raw_labels": <str>,
# }