"""Mixed dataset that combines pickscore, ocr, and geneval prompt sources.

Each sample carries a 'source' field in its metadata indicating which dataset
it came from. This determines which rewards are applicable:
  - pickscore prompts → PickScore, HPSv2, CLIPScore
  - ocr prompts      → PickScore, HPSv2, CLIPScore, OCR
  - geneval prompts   → PickScore, HPSv2, CLIPScore, GenEval
"""

import json
import os

from torch.utils.data import Dataset


class MixedPromptDataset(Dataset):
    """Dataset that mixes prompts from multiple sources with source tags."""

    def __init__(self, dataset_root, split="train", max_samples_per_source=None):
        """
        Args:
            dataset_root: Root directory containing pickscore/, ocr/, geneval/ subdirs.
            split: "train" or "test".
            max_samples_per_source: Optional dict {source: max_count} to cap samples
                per source. Useful for balancing dataset sizes.
                Example: {"pickscore": 20000, "ocr": 20000, "geneval": 20000}
        """
        if max_samples_per_source is None:
            max_samples_per_source = {}

        self.prompts = []
        self.metadatas = []

        # Load pickscore prompts
        pickscore_path = os.path.join(dataset_root, "pickscore", f"{split}.txt")
        if os.path.exists(pickscore_path):
            with open(pickscore_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            cap = max_samples_per_source.get("pickscore", len(lines))
            for p in lines[:cap]:
                self.prompts.append(p)
                self.metadatas.append({"source": "pickscore"})

        # Load OCR prompts
        ocr_path = os.path.join(dataset_root, "ocr", f"{split}.txt")
        if os.path.exists(ocr_path):
            with open(ocr_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            cap = max_samples_per_source.get("ocr", len(lines))
            for p in lines[:cap]:
                self.prompts.append(p)
                self.metadatas.append({"source": "ocr"})

        # Load GenEval prompts
        geneval_path = os.path.join(dataset_root, "geneval", f"{split}_metadata.jsonl")
        if os.path.exists(geneval_path):
            with open(geneval_path, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f if line.strip()]
            cap = max_samples_per_source.get("geneval", len(items))
            for item in items[:cap]:
                meta = dict(item)
                meta["source"] = "geneval"
                self.prompts.append(item["prompt"])
                self.metadatas.append(meta)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas
