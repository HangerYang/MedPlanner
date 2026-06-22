from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .conversation import MedicalConversation


class Qwen3Embedding:
    def __init__(self, model_name="Qwen/Qwen3-4B", device_map="cuda:0", dtype=torch.bfloat16) -> None:
        print(f"[init] Loading Qwen3 embedding model ({model_name}) on {device_map}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        self.output_dim = int(self.model.config.hidden_size)

    def _messages(self, value):
        if isinstance(value, MedicalConversation):
            return value.as_chat()
        if isinstance(value, list):
            return value
        return [{"role": "user", "content": str(value)}]

    def embed(self, value) -> torch.Tensor:
        messages = self._messages(value)
        with torch.no_grad():
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            ).to(self.model.device)
            base = getattr(self.model, "model", self.model)
            outputs = base(input_ids=input_ids)
            embedding = outputs.last_hidden_state[:, -1, :].detach().float().cpu()[0]
        return embedding
