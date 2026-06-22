import os
import re
import torch
import logging
from keys import mykey

# vLLM tensor-parallel workers default to fork(), which breaks CUDA in child processes.
# Spawn is safe; set before any `import vllm` (see vLLM troubleshooting: multiprocessing).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# A dictionary to cache models and tokenizers to avoid reloading

global models
models = {}

def log_info(message, logger_name="message_logger", print_to_std=False, mode="info"):
    logger = logging.getLogger(logger_name)
    if logger: 
        if mode == "error": logger.error(message)
        if mode == "warning": logger.warning(message)
        else: logger.info(message)
    if print_to_std: print(message + "\n")


def _chat_template_kwargs(model_name: str) -> dict:
    """Per-model overrides for tokenizer.apply_chat_template.

    Qwen3's chat template emits a `<think>...</think>` reasoning block before the
    final answer when `enable_thinking=True` (its default). For benchmarks that
    only parse the final answer this wastes ~hundreds-to-thousands of decode
    tokens per call, so we force it off by default. Set the env var
    MEDIQ_ENABLE_THINKING=1 (or true/yes) to re-enable Qwen3 reasoning.

    Other templates that don't reference `enable_thinking` silently ignore the
    kwarg.
    """
    name = (model_name or "").lower()
    kw = {}
    if "qwen" in name:
        env = os.environ.get("MEDIQ_ENABLE_THINKING", "").strip().lower()
        enable = env in ("1", "true", "yes", "on")
        kw["enable_thinking"] = enable
    return kw


def _strip_thinking(text: str) -> str:
    """Remove MedGemma-style thinking envelopes like `<unused94>...thought...<unused95>ANSWER`.
    Applied to every LLM response so downstream parsers and convo logs only see the final answer.
    """
    if not text:
        return text
    stripped = re.sub(r'<[^>]+>.*?<[^>]+>', '', text, flags=re.DOTALL).strip()
    if not stripped:
        # thinking block was cut off mid-stream (no closing tag) → keep text after the last '>'
        parts = re.split(r'>[^>]*$', text)
        stripped = parts[-1].strip() if len(parts) > 1 else text.strip()
    return stripped


def _auto_vllm_gpu_memory_utilization() -> float:
    """vLLM defaults to 0.9 × total VRAM, which fails if much VRAM is already in use (same
    process or others). cuda:0 here is the first *visible* device after CUDA_VISIBLE_DEVICES.
    """
    if not torch.cuda.is_available():
        return 0.9
    free_b, total_b = torch.cuda.mem_get_info(0)
    if total_b <= 0:
        return 0.9
    # Stay slightly under free/total so vLLM's startup check passes.
    frac = (free_b / float(total_b)) * 0.92
    return max(0.05, min(0.9, frac))


class ModelCache:
    def __init__(self, model_name, use_vllm=False, use_api=None, **kwargs):
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.use_api = use_api
        self.model = None
        self.tokenizer = None
        self.terminators = None
        self.client = None
        self.args = kwargs
        self.load_model_and_tokenizer()
    
    def load_model_and_tokenizer(self):
        if self.use_api == "openai":
            from openai import OpenAI
            self.api_account = self.args.get("api_account", "openai")
            self.client = OpenAI(api_key=mykey[self.api_account]) # Setup API key appropriately in keys.py
        elif self.use_vllm:
            try:
                from vllm import LLM
                enable_prefix_caching = self.args.get("enable_prefix_caching", True)
                tensor_parallel_size = self.args.get("tensor_parallel_size", 1)
                max_num_seqs = int(self.args.get("vllm_max_num_seqs") or self.args.get("batch_size", 256))
                cap_seq = os.environ.get("MEDIQ_VLLM_MAX_NUM_SEQS", "").strip()
                if cap_seq:
                    max_num_seqs = min(max_num_seqs, int(cap_seq))
                gpu_mem = self.args.get("gpu_memory_utilization")
                _auto_gpu = False
                if gpu_mem is None:
                    env_g = os.environ.get("MEDIQ_VLLM_GPU_MEMORY_UTILIZATION", "").strip()
                    if env_g:
                        gpu_mem = float(env_g)
                if gpu_mem is None:
                    gpu_mem = _auto_vllm_gpu_memory_utilization()
                    _auto_gpu = True
                # KV cache scales with max_model_len; models like Llama-3.1 default to 128k and OOM on one GPU.
                explicit_max_model_len = self.args.get("vllm_max_model_len") is not None
                max_model_len = self.args.get("vllm_max_model_len")
                if max_model_len is None:
                    env_m = os.environ.get("MEDIQ_VLLM_MAX_MODEL_LEN", "").strip()
                    max_model_len = int(env_m) if env_m else 8192
                max_model_len = int(max_model_len)
                mtok = self.args.get("max_tokens")
                if not explicit_max_model_len and isinstance(mtok, (int, float)) and mtok > 0:
                    floor_len = int(mtok) + 8192
                    if max_model_len < floor_len:
                        max_model_len = floor_len
                llm_kw = dict(
                    model=self.model_name,
                    enable_prefix_caching=enable_prefix_caching,
                    tensor_parallel_size=tensor_parallel_size,
                    max_num_seqs=max_num_seqs,
                    max_model_len=max_model_len,
                )
                llm_kw["gpu_memory_utilization"] = gpu_mem
                if self.args.get("vllm_enforce_eager") or os.environ.get(
                    "MEDIQ_VLLM_ENFORCE_EAGER", ""
                ).strip().lower() in ("1", "true", "yes"):
                    llm_kw["enforce_eager"] = True
                swap_s = os.environ.get("MEDIQ_VLLM_SWAP_SPACE_GB", "").strip()
                if swap_s:
                    llm_kw["swap_space"] = float(swap_s)
                log_info(
                    f"[vLLM] {self.model_name}: max_model_len={max_model_len}, max_num_seqs={max_num_seqs}, "
                    f"gpu_memory_utilization={gpu_mem:.4f}"
                    + (" (gpu util from free VRAM)" if _auto_gpu else ""),
                    print_to_std=True,
                )
                self.model = LLM(**llm_kw)
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.terminators = self._get_terminators()
            except Exception as e:
                raise RuntimeError(f"[{self.model_name}] vLLM failed to load: {e}")
        if not self.use_vllm and self.use_api != "openai":
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
            try:
                self.tokenizer = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto")
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto")
            self.model.eval()
            # Processor wraps a tokenizer; fall back to the inner tokenizer for token attrs
            _tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            _tok.pad_token = _tok.eos_token
            _tok.pad_token_id = _tok.eos_token_id
            self.terminators = self._get_terminators()
    
    def _get_terminators(self):
        _tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        terminators = [_tok.eos_token_id]
        for token in ["<|eot_id|>", "<end_of_turn>", "<|im_end|>", "</s>"]:
            tid = _tok.convert_tokens_to_ids(token)
            if tid and tid != _tok.unk_token_id:
                terminators.append(tid)
        return list(set(terminators))

    def generate(self, messages):

        self.temperature = self.args.get("temperature", 0.6)
        self.max_tokens = self.args.get("max_tokens", 256)
        self.top_p = self.args.get("top_p", 0.9)
        self.top_logprobs = self.args.get("top_logprobs", 0)

        if self.use_api == "openai":
            result = self.openai_generate(messages)
        elif self.use_vllm:
            result = self.vllm_generate(messages)
        else:
            result = self.huggingface_generate(messages)
        return result
    
    def _to_content_list(self, messages):
        """Convert plain string content to list-of-dicts format expected by processors like MedGemma."""
        out = []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            out.append({"role": m["role"], "content": content})
        return out

    def huggingface_generate(self, messages):
        tmpl_kwargs = _chat_template_kwargs(self.model_name)
        inputs = self.tokenizer.apply_chat_template(
            self._to_content_list(messages), add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
            **tmpl_kwargs,
        ).to(self.model.device)
        inputs = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)

        raw_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        log_info(f"[{self.model_name}][RAW OUTPUT]:\n{raw_text}")
        print(f"[LLM RAW OUTPUT]:\n{raw_text}\n")
        response_text = _strip_thinking(raw_text)
        usage = {"input_tokens": input_len, "output_tokens": outputs.shape[-1] - input_len}
        log_info(f"[{self.model_name}][PARSED OUTPUT]: {response_text}")
        return response_text, None, usage
        
    def vllm_generate(self, messages):
        tmpl_kwargs = _chat_template_kwargs(self.model_name)
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **tmpl_kwargs
            )
        except Exception:
            log_info(f"[{self.model_name}]: Could not apply chat template to messages.", mode="warning")
            inputs = "\n\n".join([m['content'] for m in messages])
            # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        from vllm import SamplingParams
        frequency_penalty = self.args.get("frequency_penalty", 0)
        presence_penalty = self.args.get("presense_penalty", 0)
        sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p, logprobs=self.top_logprobs, 
                                        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        
        outputs = self.model.generate(inputs, sampling_params)
        raw_text = outputs[0].outputs[0].text
        response_text = _strip_thinking(raw_text)
        logprobs = outputs[0].outputs[0].cumulative_logprob
        # TODO: If top_logprobs > 0, return logprobs of generation
        # if self.top_logprobs > 0: logprobs = outputs[0].outputs[0].logprobs
        usage = {"input_tokens": len(outputs[0].prompt_token_ids), "output_tokens": len(outputs[0].outputs[0].token_ids)}
        output_dict = {'response_text': response_text, 'usage': usage}

        log_info(f"[{self.model_name}][OUTPUT]: {output_dict}")
        return response_text, logprobs, usage

    def openai_generate(self, messages):
        if self.top_logprobs == 0:
            response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p
                    )
        else:
            response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        logprobs=True, 
                        top_logprobs=self.top_logprobs
                    )
        
        num_input_tokens = response["usage"]["prompt_tokens"]
        num_output_tokens = response["usage"]["completion_tokens"]
        response_text = response.choices[0].text.strip()
        log_probs = response.choices[0].logprobs.top_logprobs if self.top_logprobs > 0 else None
        
        log_info(f"[{self.model_name}][OUTPUT]: {response}")
        return response_text, log_probs, {"input_tokens": num_input_tokens, "output_tokens": num_output_tokens}


def get_response(messages, model_name, use_vllm=False, use_api=None, **kwargs):
    if 'gpt' in model_name or 'o1' in model_name: use_api = "openai"
    
    model_cache = models.get(model_name, None)
    if model_cache is None:
        model_cache = ModelCache(model_name, use_vllm=use_vllm, use_api=use_api, **kwargs)
        models[model_name] = model_cache
    
    return model_cache.generate(messages)
