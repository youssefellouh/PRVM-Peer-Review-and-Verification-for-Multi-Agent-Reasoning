import logging
import time
import os
import json
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
                # reasoning_effort= 'none'
                reasoning= {
                    'effort': 'none'
                }
            )
            end_time = time.time()
            inference_time = end_time - start_time
            time.sleep(0.5)  # Add delay to avoid rate limiting
            return completion, inference_time  # Return tuple instead of assigning attribute
        except Exception as e:
            logging.warning(f"retrying due to an error: {e}")
            time.sleep(20)

def read_jsonl(path: str):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh.readlines() if line]
def single_agent(client, args):
    model_name = getattr(args, 'model', None) or getattr(config, 'MODEL_NAME', None)
    logging.info(f'Using model: {model_name}')
    if not args.reload_data:
        generated_description = []
    else:
        check_dirs_files(dirs=[], files=[args.output_file, ])
        with open(args.output_file, 'r', encoding='utf-8') as f:
            generated_description = json.load(f)
    generated_len = len(generated_description)
    if generated_len:
        logging.info(f'reload from: {args.output_file}')
        logging.info(f'reload data num: {generated_len}')

    all_datas = read_jsonl(args.task_file)
    for i, data in enumerate(tqdm(all_datas)):

        if args.reload_data and i < generated_len:
            continue

        question = data['question']
        answer = data['answer']

        if args.task in ['GSM8K']:  # number
            content = """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(question)
        elif args.task in ['ARC-c','MMLU']:  # option
            content = "Can you answer the following question as accurately as possible? {} Explain your answer, putting the answer in the form (X) at the end of your response.".format(question)
        elif args.task in ['StrategyQA', ]:  # yes or no
            content = "Can you answer the following question as accurately as possible? {} Explain your answer, your answer should be Yes or No at the end of your response.".format(question)
        else:
            raise Exception('failed to construct question, unknown task!')
            
        agent_contexts = [[{"role": "user", "content": content}] for _ in range(args.agent_num)]  # single agent only
        inference_times = []
        token_usages = []

        for agent_context in agent_contexts:
            # Unpack returned completion and inference_time
            completion, inference_time = generate_answer(client, agent_context, model_name)
            inference_times.append(inference_time)
            
            assistant_message = construct_assistant_message(completion)
            agent_context.append(assistant_message)
            
            # Store token usage safely
            usage = getattr(completion, 'usage', None)
            if usage:
                token_usage = {
                    'prompt_tokens': getattr(usage, 'prompt_tokens', 0) if not isinstance(usage, dict) else usage.get('prompt_tokens', 0),
                    'completion_tokens': getattr(usage, 'completion_tokens', 0) if not isinstance(usage, dict) else usage.get('completion_tokens', 0),
                    'total_tokens': getattr(usage, 'total_tokens', 0) if not isinstance(usage, dict) else usage.get('total_tokens', 0)
                }
                token_usages.append(token_usage)
            else:
                token_usages.append(None)

        generated_description.append({
            'question': question,
            'answer': answer,
            'agent_contexts': agent_contexts,
            'inference_times': inference_times,
            'token_usages': token_usages,
        })

        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(generated_description, f, ensure_ascii=False)

    # Display generation metrics
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
        
        print("\n--- Single Agent Generation Metrics ---")
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
    args_str = f'\n--------------- single agent parameters ---------------\n'
    for k, v in args.__dict__.items():
        args_str += f'{k} = {v}\n'
    args_str += f'-------------------------------------------------------'
    logging.info(args_str)

if __name__ == "__main__":
    args = params.single_agent_args()
    log_param(args)

    output_task_dir = os.path.dirname(args.output_file)
    check_dirs_files(dirs=[args.dataset_dir, args.output_dir, output_task_dir], files=[args.task_file, ])

    api_key = config.OPENROUTER_API_KEY or os.getenv('OPENROUTER_API_KEY', '')
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Set it in the environment.")

    with OpenRouter(api_key=api_key) as client:
        single_agent(client, args)