from __future__ import annotations

import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "Complete the given Python function correctly. Return only the code that should be "
    "appended to the prompt. Do not use Markdown fences or explain the answer."
)


def clean_completion(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", str(text), flags=re.DOTALL)
    text = re.sub(r"^\s*```(?:python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.rstrip()
    if text and not text.splitlines()[0].startswith((" ", "\t")):
        text = "\n".join(("    " + line) if line.strip() else line for line in text.splitlines())
    return text + "\n"


class QwenCodeModel:
    def __init__(self, model_name: str, device: str, enable_thinking: bool = False, thinking_budget: int = 0) -> None:
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        print(f"[init] Loading {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

    @property
    def device(self):
        return self.model.device

    def _generation_tokens(self, prompt: str):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        template_kwargs: dict = {"enable_thinking": self.enable_thinking}
        if self.enable_thinking and self.thinking_budget > 0:
            template_kwargs["thinking_budget"] = self.thinking_budget
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            return_attention_mask=True,
            **template_kwargs,
        ).to(self.device)

    def generate_greedy(self, prompt: str, max_new_tokens: int) -> str:
        tokens = self._generation_tokens(prompt)
        input_len = tokens["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]
        return clean_completion(self.tokenizer.decode(output[input_len:], skip_special_tokens=True))

    def generate_beams(
        self,
        prompt: str,
        num_candidates: int,
        max_new_tokens: int,
        diversity_penalty: float,
        repetition_penalty: float,
    ) -> list[str]:
        tokens = self._generation_tokens(prompt)
        input_len = tokens["input_ids"].shape[-1]
        kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_candidates,
            num_beam_groups=num_candidates,
            num_return_sequences=num_candidates,
            diversity_penalty=diversity_penalty,
            repetition_penalty=repetition_penalty,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        with torch.inference_mode():
            try:
                outputs = self.model.generate(
                    **tokens,
                    **kwargs,
                    custom_generate="transformers-community/group-beam-search",
                    trust_remote_code=True,
                )
            except Exception as error:
                print(f"[warn] custom group beam search unavailable ({error}); using built-in beam search.")
                outputs = self.model.generate(**tokens, **kwargs)
        completions: list[str] = []
        for output in outputs:
            completion = clean_completion(self.tokenizer.decode(output[input_len:], skip_special_tokens=True))
            if completion not in completions:
                completions.append(completion)
        return completions

    def embed_turn(self, text: str) -> np.ndarray:
        messages = [{"role": "user", "content": str(text)}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            base = getattr(self.model, "model", self.model)
            outputs = base(input_ids=input_ids)
        return outputs.last_hidden_state[:, -1, :].detach().float().cpu().numpy()[0]
