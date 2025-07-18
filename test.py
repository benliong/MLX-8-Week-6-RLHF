from datasets import load_dataset


from huggingface_hub import HfApi, get_session
from huggingface_hub.utils import hf_raise_for_status
from urllib.parse import quote

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

HfApi.get_paths_info = _patched_get_paths_info   # hot-patch

ds = load_dataset("CarperAI/openai_summarize_tldr", split="train")
print(len(ds), ds[0])