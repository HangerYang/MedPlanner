import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Run the benchmark with specified configurations.")
    parser.add_argument('--expert_module', type=str, default='expert', help='file name where the expert class is implemented.')
    parser.add_argument('--expert_class', type=str, required=True, help='Expert class name to use for the benchmark.')
    parser.add_argument('--expert_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Expert model name to use for the benchmark, can be a local model or a Huggingface model.')
    parser.add_argument('--expert_model_question_generator', type=str, default=None, help='Separate model for follow-up question generation. Defaults to expert_model if not set.')
    
    parser.add_argument('--patient_module', type=str, default='patient', help='file name where the patient class is implemented.')
    parser.add_argument('--patient_class', type=str, required=True, help='Patient class name to use for the benchmark.')
    parser.add_argument('--patient_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Patient model name to use for the benchmark, can be a local model or a Huggingface model.')
    
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the development data files.')
    parser.add_argument('--dev_filename', type=str, required=True, help='Filename for development data.')

    parser.add_argument('--output_filename', type=str, default="results.jsonl")
    parser.add_argument('--max_examples', type=int, default=-1, help='Max number of examples to run. -1 means all.')
    parser.add_argument('--num_shards', type=int, default=1, help='Total number of data-parallel shards.')
    parser.add_argument('--shard_idx', type=int, default=0, help='Zero-based index of this data-parallel shard.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output file.')

    parser.add_argument("--max_questions", type=int, default=30)
    parser.add_argument(
        "--branch_depth",
        type=int,
        default=0,
        help="If >0, use binary branching question exploration for this many levels.",
    )
    parser.add_argument(
        "--branch_top_k",
        type=int,
        default=2,
        help="Number of follow-up questions to generate per branch split.",
    )

    parser.add_argument('--log_filename', type=str, default=None, help='Filename for logging general benchmark results.')
    parser.add_argument('--history_log_filename', type=str, default=None, help='Filename for logging interaction history, will not log if None.')
    parser.add_argument('--detail_log_filename', type=str, default=None, help='Filename for logging detailed prompts and response on abstention, will not log if None.')
    parser.add_argument('--message_log_filename', type=str, default=None, help='Filename for logging messages passed into API calls, will not log if None.')
    parser.add_argument('--convo_log_filename', type=str, default=None, help='Filename (.txt) for logging full per-case text reports (confidence, shadow answer, etc.), will not log if None.')
    parser.add_argument('--doctor_log_filename', type=str, default=None, help='Filename (.txt) for logging only what the doctor sees (initial info, Q&A history, final answer), will not log if None.')

    parser.add_argument('--rationale_generation', action='store_true', help='Generate rationales for the choices.')
    parser.add_argument('--self_consistency', type=int, default=1, help='Number of times to run the self-consistency check.')
    parser.add_argument('--abstain_threshold', type=float, default=0.8, help='Threshold for abstaining from making a choice.')
    parser.add_argument('--independent_modules', action='store_true', help='Cognitive modules within the Expert dont see previous convo.')
    parser.add_argument('--option_mode', type=str, default='yes-option',
        choices=['yes-option', 'no-option', 'option-in-the-end'],
        help='yes-option: expert always sees A/B/C/D and picks a letter. '
             'no-option: expert never sees options, answers in \\box{}, judged by patient model. '
             'option-in-the-end: no options during Q&A turns; options shown only at the final decision.')

    parser.add_argument('--use_vllm', action='store_true', help='Use the VLLM model for generating responses.')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of GPUs for vLLM tensor parallelism.')
    parser.add_argument('--batch_size', type=int, default=256, help='vLLM max_num_seqs (batch size).')
    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=None,
        help='vLLM fraction of total VRAM (0-1). If omitted, helper infers from free VRAM. Env MEDIQ_VLLM_GPU_MEMORY_UTILIZATION overrides.',
    )
    parser.add_argument(
        '--vllm_max_model_len',
        type=int,
        default=8192,
        help='vLLM max_model_len (KV cache). Default 8192 avoids 128k-context OOM on one GPU; raise for long context.',
    )
    parser.add_argument(
        '--vllm_max_num_seqs',
        type=int,
        default=None,
        help='vLLM max_num_seqs; if omitted, uses --batch_size. Lower (e.g. 8–32) saves VRAM. Env MEDIQ_VLLM_MAX_NUM_SEQS caps further.',
    )
    parser.add_argument(
        '--vllm_enforce_eager',
        action='store_true',
        help='vLLM enforce_eager (less peak memory, slower). Or set MEDIQ_VLLM_ENFORCE_EAGER=1.',
    )
    parser.add_argument('--use_api', type=str, default=None, help='Use an API for generating responses.', choices=['openai']) # compatible with the OpenAI API for now
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for sampling from the model.')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top p value for nucleus sampling.')
    parser.add_argument('--max_tokens', type=int, default=256, help='Maximum number of tokens to generate.')
    parser.add_argument('--top_logprobs', type=int, default=0, help='Number of top logprobs to return.')
    parser.add_argument('--api_account', type=str, default="mediQ", help='API keys are stored in keys.py, api_account is the name of the key.')
    
    args =  parser.parse_args()

    for f in [args.log_filename, args.history_log_filename, args.detail_log_filename, args.message_log_filename]:
        if f and os.path.dirname(f): os.makedirs(os.path.dirname(f), exist_ok=True)
    return args
