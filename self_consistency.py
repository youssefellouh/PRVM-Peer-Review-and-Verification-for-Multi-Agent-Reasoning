import logging
import os
import re
import time
import json
from collections import Counter
from typing import List, Optional
import numpy as np
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from data_proc import check_dirs_files
from openrouter import OpenRouter
import params
import config

def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}

def generate_answer(client, answer_context, model_name):
    while True:
        try:
            start_time = time.time()
            completion = client.chat.send(
                model=model_name,
                messages=answer_context,
                stream=False,
                reasoning={"effort": "none"},
            )
            end_time = time.time()
            inference_time = end_time - start_time
            time.sleep(0.5)
            return completion, inference_time
        except Exception as e:
            logging.warning(f"retrying due to an error: {e}")
            time.sleep(20)

def read_jsonl(path: str):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def build_prompt(task: str, question: str) -> str:
    if task in ['GSM8K']:
        return (
            "Can you solve the following math problem? {} Explain your reasoning. "
            "Your final answer should be a single numerical number, in the form \\boxed{{answer}}, "
            "at the end of your response.".format(question)
        )
    if task in ['ARC-c','MMLU']:
        return (
            "Can you answer the following question as accurately as possible? {} "
            "Explain your answer, putting the answer in the form (X) at the end of your response."
            .format(question)
        )
    if task in ['StrategyQA']:
        return (
            "Can you answer the following question as accurately as possible? {} "
            "Explain your answer, your answer should be Yes or No at the end of your response."
            .format(question)
        )
    raise Exception('failed to construct question, unknown task!')

def parse_pred_answer(input_str: Optional[str], task: str):
    if input_str is None:
        return None
    if task in ['GSM8K']:
        pattern = r"\{([-0-9.,$]*)\}"
        matches = re.findall(pattern, input_str)
        if not matches:
            matches = re.findall(r'-?\d+(?:\.\d+)?', input_str)
        for match_str in matches[::-1]:
            solution = re.sub(r"[^-0-9.]", "", match_str)
            if solution:
                return solution
        return None
    if task in ['ARC-c','MMLU']:
        pattern = r'\b([A-E])\b|\(([A-E])\)'
        matches = re.findall(pattern, input_str)
        for match_tuple in matches[::-1]:
            solution = next((m for m in match_tuple if m), None)
            if solution and solution.upper() != 'X':
                return solution.upper()
        return None
    if task in ['StrategyQA']:
        pattern = r'\b(YES|Yes|yes|NO|No|no)\b'
        matches = re.findall(pattern, input_str)
        for match_str in matches[::-1]:
            if match_str in ['Yes', 'yes', 'YES']:
                return 'Yes'
            if match_str in ['No', 'no', 'NO']:
                return 'No'
        return None
    return None

def parse_gt_answer(input_str: Optional[str], task: str):
    if input_str is None:
        return None
    if task in ['GSM8K']:
        answer = input_str.split('####')[-1].replace(',', '').strip()
        return answer or None
    return input_str

def most_frequent(lst: List[Optional[str]]):
    if not lst:
        return None
    return Counter(lst).most_common(1)[0][0]

def compute_accuracy(gt: Optional[str], pred_solutions, task: str) -> int:
    gt_answer = parse_gt_answer(gt, task)
    if gt_answer is None:
        return 0
    if isinstance(pred_solutions, list):
        pred_answers = []
        for pred_solution in pred_solutions:
            pred_answer = parse_pred_answer(pred_solution, task)
            if pred_answer:
                pred_answers.append(pred_answer)
        pred_answer = most_frequent(pred_answers)
    else:
        pred_answer = parse_pred_answer(pred_solutions, task)

    if pred_answer is None:
        return 0
    if task in ['GSM8K']:
        try:
            return 1 if float(gt_answer) == float(pred_answer) else 0
        except Exception:
            return 0
    if task in ['ARC-c', 'StrategyQA','MMLU']:
        return 1 if gt_answer == pred_answer else 0
    return 0

def self_consistency(client, args, model_name):
    if not args.reload_data:
        generated_description = []
    else:
        check_dirs_files(dirs=[], files=[args.output_file])
        with open(args.output_file, 'r', encoding='utf-8') as f:
            generated_description = json.load(f)
    generated_len = len(generated_description)
    if generated_len:
        logging.info(f'reload from: {args.output_file}')
        logging.info(f'reload data num: {generated_len}')

    all_datas = read_jsonl(args.task_file)
    sample_num = getattr(args, 'sample_num', getattr(args, 'agent_num', 3))

    for i, data in enumerate(tqdm(all_datas)):
        if args.reload_data and i < generated_len:
            continue

        question = data['question']
        answer = data['answer']
        prompt = build_prompt(args.task, question)

        agent_contexts = []
        inference_times = []
        token_usages = []
        sampled_answers = []
        parsed_answers = []

        for _ in range(sample_num):
            agent_context = [{"role": "user", "content": prompt}]
            completion, inference_time = generate_answer(client, agent_context, model_name)
            inference_times.append(inference_time)

            assistant_message = construct_assistant_message(completion)
            agent_context.append(assistant_message)
            agent_contexts.append(agent_context)

            response_text = assistant_message['content']
            sampled_answers.append(response_text)
            parsed_answer = parse_pred_answer(response_text, args.task)
            parsed_answers.append(parsed_answer)
            usage = getattr(completion, 'usage', None)
            if usage:
                token_usage = {
                    'prompt_tokens': getattr(usage, 'prompt_tokens', 0) if not isinstance(usage, dict) else usage.get('prompt_tokens', 0),
                    'completion_tokens': getattr(usage, 'completion_tokens', 0) if not isinstance(usage, dict) else usage.get('completion_tokens', 0),
                    'total_tokens': getattr(usage, 'total_tokens', 0) if not isinstance(usage, dict) else usage.get('total_tokens', 0),
                }
                token_usages.append(token_usage)
            else:
                token_usages.append(None)

        majority_answer = most_frequent(parsed_answers)
        accuracy = compute_accuracy(answer, sampled_answers, args.task)
        generated_description.append({
            'question': question,
            'answer': answer,
            'agent_contexts': agent_contexts,
            'inference_times': inference_times,
            'token_usages': token_usages,
            'sampled_answers': sampled_answers,
            'parsed_answers': parsed_answers,
            'majority_answer': majority_answer,
            'accuracy': accuracy,
            'sample_num': sample_num,
        })
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(generated_description, f, ensure_ascii=False)
    if generated_description:
        total_tokens_list = []
        all_inference_times = []
        for entry in generated_description:
            if 'token_usages' in entry and entry['token_usages']:
                for usage in entry['token_usages']:
                    if usage and 'total_tokens' in usage:
                        total_tokens_list.append(int(usage['total_tokens']))
            if 'inference_times' in entry and isinstance(entry['inference_times'], list):
                for t in entry['inference_times']:
                    try:
                        all_inference_times.append(float(t))
                    except Exception:
                        continue
        print("\n--- Self-Consistency Generation Metrics ---")
        if total_tokens_list:
            token_mean = np.mean(total_tokens_list)
            token_total = sum(total_tokens_list)
            print(f"Token Usage: mean={token_mean:.1f} tokens/call, total={token_total} tokens")
        if all_inference_times:
            time_mean = np.mean(all_inference_times)
            time_std = np.std(all_inference_times)
            time_count = len(all_inference_times)
            print(f"Inference time: mean={time_mean:.4f}s, std={time_std:.4f}s, calls={time_count}")

def log_param(args):
    args_str = f'\n--------------- self consistency parameters ---------------\n'
    for k, v in args.__dict__.items():
        args_str += f'{k} = {v}\n'
    args_str += f'-------------------------------------------------------'
    logging.info(args_str)
def main():
    args = params.self_consistency_args()
    log_param(args)
    output_task_dir = os.path.dirname(args.output_file)
    check_dirs_files(dirs=[args.dataset_dir, args.output_dir, output_task_dir], files=[args.task_file])
    model_name = getattr(args, 'model', None) or getattr(config, 'MODEL_NAME', None)
    logging.info(f'Using model: {model_name}')
    api_key = config.OPENROUTER_API_KEY or os.getenv('OPENROUTER_API_KEY', '')
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Set it in the environment.")
    with OpenRouter(api_key=api_key) as client:
        self_consistency(client, args, model_name)

if __name__ == '__main__':
    main()
