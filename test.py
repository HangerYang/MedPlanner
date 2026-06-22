import os

# tensor_parallel_size > 1 uses worker subprocesses. Default "fork" can break CUDA
# in those children (fork after CUDA runtime is unsafe). "spawn" fixes it.
# With spawn, the engine re-imports this file; keep vLLM usage inside main().
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def main():
    from vllm import LLM, SamplingParams

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    llm = LLM(
        model=model_name,
        tensor_parallel_size=4,
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
    )

    prompts = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
        "<|im_start|>assistant\n"
    ]

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(output.outputs[0].text)


if __name__ == "__main__":
    main()
