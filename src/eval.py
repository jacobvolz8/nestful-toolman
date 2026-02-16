import os, json, argparse
from utils import *
from instruct_data_prep import get_instruct_data
import gc

def _proxy_generate(prompts, *, proxy_url, model_fqn, temperature, max_tokens, timeout=600):
    import urllib.request
    import urllib.error

    endpoint = proxy_url.rstrip("/") + "/generate"
    payload = {
        "prompts": prompts,
        "model": model_fqn,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "system_prompt": "",
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            res_body = res.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM proxy HTTP {e.code}: {err_body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"LLM proxy request failed: {e}")

    parsed = json.loads(res_body)
    texts = parsed.get("texts") or []
    if not texts:
        raise RuntimeError(f"LLM proxy response had no texts. Keys: {list(parsed.keys())}")
    return [t.strip() for t in texts]

def _proxy_nestful(query, tools, *, proxy_url, model_fqn, temperature, max_tokens, enable_ptc=False, tool_choice="required", system_prompt="", js_extract_timeout_ms=None, timeout=600):
    import urllib.request
    import urllib.error

    endpoint = proxy_url.rstrip("/") + "/nestful"
    payload = {
        "bellman_model": model_fqn,
        "query": query,
        "tools": tools,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "system_prompt": system_prompt or "",
        "enable_ptc": bool(enable_ptc),
        "tool_choice": tool_choice or "required",
    }
    if js_extract_timeout_ms is not None:
        payload["js_extract_timeout_ms"] = int(js_extract_timeout_ms)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            res_body = res.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NESTFUL proxy HTTP {e.code}: {err_body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"NESTFUL proxy request failed: {e}")

    parsed = json.loads(res_body)
    gen_text = parsed.get("generated_text")
    if gen_text is None:
        return "[]"
    if not isinstance(gen_text, str):
        gen_text = json.dumps(gen_text)
    gen_text = gen_text.strip()
    if gen_text == "null":
        return "[]"
    return gen_text

def eval_code(args):
    print("### Loading Data...")
    
    data = read_jsonlines(args.dataset)

    # If you want to call the Go proxy's /nestful endpoint, we should send the *raw*
    # sample input + tools (not the ICL-formatted prompt used for /generate).
    if args.llm_proxy_url and getattr(args, "llm_proxy_endpoint", "generate") == "nestful":
        print("### Using local NESTFUL proxy (/nestful)...")
        model_fqn = args.llm_proxy_model or args.model

        output_list = []
        count_total = len(data)
        for idx, sample in enumerate(data):
            if (idx % max(1, int(args.batch_size))) == 0:
                print(f"### At sample {idx + 1} out of {count_total}...")

            query = sample.get("input", "")
            tools_spec = sample.get("tools") or []
            gen_text = _proxy_nestful(
                query,
                tools_spec,
                proxy_url=args.llm_proxy_url,
                model_fqn=model_fqn,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                enable_ptc=args.enable_ptc,
                tool_choice=args.tool_choice,
                system_prompt=args.system_prompt,
                js_extract_timeout_ms=args.js_extract_timeout_ms,
            )

            # Keep output format consistent with the rest of the codebase (strings in jsonl).
            output_list.append(
                {
                    "sample_id": sample.get("sample_id", ""),
                    "input": query,
                    "output": json.dumps(sample.get("output", [])),
                    "gold_answer": json.dumps(sample.get("gold_answer")),
                    "tools": json.dumps(tools_spec),
                    "generated_text": gen_text,
                }
            )

        print("### Saving...")
        save_path = os.path.join(args.save_directory, f"nestful_{args.icl_count}", args.model_name, "output.jsonl")
        print(f"### Save Path: {save_path}")
        os.makedirs(os.path.join(args.save_directory, f"nestful_{args.icl_count}", args.model_name), exist_ok=True)
        write_jsonlines(output_list, save_path)

        print(f"### DONE...!!!")
        return

    for i in range(len(data)):
        data[i]["tools"] = json.dumps(data[i]["tools"])
        data[i]["gold_answer"] = json.dumps(data[i]["gold_answer"])
        data[i]["output"] = json.dumps(data[i]["output"])

    print("### Preparing Instruct Data...")
    instruct_data = get_instruct_data(data, args.model, args.model_name, args.icl_count)

    prompts = [sample["input"] for sample in instruct_data]

    print("### Starting Generation...")
    response, output_list = [], []

    if args.llm_proxy_url:
        print("### Using local LLM proxy...")
        model_fqn = args.llm_proxy_model or args.model
        count_total_batches = -(-len(prompts) // args.batch_size)
        for idx in range(0, len(prompts), args.batch_size):
        #for idx in range(5):
            print(f"### At batch {idx // args.batch_size + 1} out of {count_total_batches} batches...")
            prompt_batch = prompts[idx : idx + args.batch_size]
            batch_texts = _proxy_generate(
                prompt_batch,
                proxy_url=args.llm_proxy_url,
                model_fqn=model_fqn,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            response.extend(batch_texts)
    else:
        print("### Loading Model (vLLM)...")
        import torch
        from vllm import LLM, SamplingParams
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )

        llm = LLM(
            model=args.model,
            tensor_parallel_size=torch.cuda.device_count(),
            disable_custom_all_reduce=True,
        )

        sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

        count_total_batches = -(-len(prompts) // args.batch_size)
        for idx in range(0, len(prompts), args.batch_size):
            print(f"### At batch {idx // args.batch_size + 1} out of {count_total_batches} batches...")
            prompt_batch = prompts[idx : idx + args.batch_size]
            batch_output = llm.generate(prompt_batch, sampling_params)

            for output in batch_output:
                response.append(output.outputs[0].text.strip())

        # unload model from GPU
        destroy_model_parallel()
        destroy_distributed_environment()
        del llm.llm_engine.model_executor
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    for idx in range(len(response)):
        temp = instruct_data[idx]
        temp["generated_text"] = response[idx]
        output_list.append(temp)

    print("### Saving...")
    save_path = os.path.join(args.save_directory, f"nestful_{args.icl_count}", args.model_name, "output.jsonl")
    print(f"### Save Path: {save_path}")
    os.makedirs(os.path.join(args.save_directory, f"nestful_{args.icl_count}", args.model_name), exist_ok=True)
    write_jsonlines(output_list, save_path)

    print(f"### DONE...!!!")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    default_dataset = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data_v2", "nestful_data.jsonl")
    )

    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini-nestful",
    )
    parser.add_argument(
        "--llm_proxy_url",
        type=str,
        default="http://localhost:8091",
        help="Local LLM proxy URL, e.g. http://localhost:8091",
    )
    parser.add_argument(
        "--llm_proxy_model",
        type=str,
        default="",
        help="Model as provider/name passed to proxy. Defaults to --model.",
    )
    parser.add_argument(
        "--llm_proxy_endpoint",
        type=str,
        default="nestful",
        choices=["generate", "nestful"],
        help="Which proxy endpoint to call: /generate or /nestful",
    )
    parser.add_argument(
        "--enable_ptc",
        type=lambda x: str(x).lower() in ("1", "true", "t", "yes", "y"),
        default=True,
        help="If true, sets enable_ptc=true for /nestful calls",
    )
    parser.add_argument(
        "--tool_choice",
        type=str,
        default="required",
        choices=["auto", "required", "none"],
        help="tool_choice for /nestful calls",
    )
    parser.add_argument("--save_directory", type=str, default="results")
    parser.add_argument("--dataset", type=str, default=default_dataset)
    parser.add_argument("--icl_count", default=3, type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--js_extract_timeout_ms",
        type=int,
        default=5000,
        help="Timeout for JS extraction when enable_ptc=true",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "Inside code_execution, you MUST use the provided tools for all arithmetic; "
            "do not use + - * / except for loop counters; call add/multiply/divide/... "
            "for each step; returning without tool calls is invalid. "
            "Do not use +-*/ for math. Use add/subtract/multiply/divide for ALL arithmetic. "
            "Do not nest tool calls. Always assign: const a = divide({...}); then pass a.result vidare. "
            "Every tool call must include every required arg (arg_0, arg_1, ..., arg_n)."
        ),
    )

    args = parser.parse_args()
    print(args)
    eval_code(args)
