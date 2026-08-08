import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from data_proc import check_dirs_files
import time
import os
import json
import numpy as np
from openrouter import OpenRouter
from tqdm import tqdm
import jsonlines
import params
import config

def read_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)
def write_json(output_path, output_data):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False)

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
                reasoning={
                    'effort': 'none'
                }
            )
            end_time = time.time()
            inference_time = end_time - start_time
            return completion, inference_time
        except Exception as e:
            logging.warning(f"retrying due to an error: {e}")
            time.sleep(60)

def read_jsonl(path: str):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def get_token_usage(completion):
    if hasattr(completion, 'usage') and completion.usage:
        usage = completion.usage
        if isinstance(usage, dict):
            return {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
        return {
            'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
            'completion_tokens': getattr(usage, 'completion_tokens', 0),
            'total_tokens': getattr(usage, 'total_tokens', 0)
        }
    return None

def feedback(client, args, model_name):
    if not args.reload_data:
        generated_description = []
    else:
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
        elif args.task in ['StrategyQA']:  # yes or no
            content = "Can you answer the following question as accurately as possible? {} Explain your answer, your answer should be Yes or No at the end of your response.".format(question)
        else:
            raise Exception('failed to construct question, unknown task!')
        agent_contexts = [[{"role": "user", "content": content}] for _ in range(args.agent_num)]

        agent_init_ans = None  # agent initial answer
        agent_feedbacks = [[] for _ in range(args.agent_num)]  # agent feedback
        inference_times = []  # collect all inference times
        token_usages = []
        for round_num in range(args.rounds):

            if round_num == 1:  # update agent initial answer
                agent_init_ans = [agent_contexts[k][1]['content'] for k in range(args.agent_num)]

            for j, agent_context in enumerate(agent_contexts):
                if round_num == 0:  # ROUND 0: generate initial answer
                    completion, inference_time = generate_answer(client, agent_context, model_name)
                    inference_times.append(inference_time)
                    token_usages.append(get_token_usage(completion))
                    assistant_message = construct_assistant_message(completion)
                    agent_context.append(assistant_message)
                elif round_num == 1:  # ROUND 1: give feedback to each other
                    ans_to_add = [k for k in range(args.agent_num) if k != j]
                    for index in ans_to_add:
                        init_ans = agent_init_ans[index]
                        content = f"Here is a solution from another agent: \n\n {init_ans}\n\n Please examine this agent's reasoning process step by step and offer feedback on its reasoning."
                        agent_context.append({"role": "user", "content": content})
                        completion, inference_time = generate_answer(client, agent_context, model_name)
                        inference_times.append(inference_time)
                        token_usages.append(get_token_usage(completion))
                        assistant_message = construct_assistant_message(completion)
                        agent_context.append(assistant_message)
                        agent_feedbacks[index].append(assistant_message)
                elif round_num == 2:  # ROUND 2: base on the initial answer and other agents' feedback, update final answer
                    agent_feedback = agent_feedbacks[j]
                    agent_num_dict = {1: "one", 2: "two", 3: "three", 4: "four"}
                    content = f"Here are the feedbacks for your solution from the above {agent_num_dict[args.agent_num - 1]} agents:\n\n "
                    for feedback in agent_feedback:
                        content += f"One agent feedback: {feedback['content']} \n\n "

                    if args.task in ['GSM8K']:  # number
                        content += f"Using other agents' solutions and feedbacks as additional information, " \
                                   f"can you provide your answer to the math problem? \n " \
                                   f"The original math problem is {question}. " \
                                   f"Your final answer should be a single numerical number, " \
                                   f"in the form \\boxed{{answer}}, at the end of your response."
                    elif args.task in ['ARC-c','MMLU']:  # option
                        content += f"Using the reasoning from other agents as additional advice, " \
                                   f"can you give an updated answer? Examine your solution and other agents' feedback step by step. " \
                                   f"Put your answer in the form (X) at the end of your response."
                    elif args.task in ['StrategyQA']:  # yes or no
                        content += f"Using the reasoning from other agents as additional advice, " \
                                   f"can you give an updated answer? Examine your solution and other agents' feedback step by step. " \
                                   f"Your answer should be Yes or No at the end of your response."
                    else:
                        raise Exception('failed to construct question, unknown task!')

                    agent_context.append({"role": "user", "content": content})
                    completion, inference_time = generate_answer(client, agent_context, model_name)
                    inference_times.append(inference_time)
                    token_usages.append(get_token_usage(completion))
                    assistant_message = construct_assistant_message(completion)
                    agent_context.append(assistant_message)

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
        
        print("\n--- Feedback Generation Metrics ---")
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
    args_str = f'\n---------------- cross critique parameters ----------------\n'
    for k, v in args.__dict__.items():
        args_str += f'{k} = {v}\n'
    args_str += f'-------------------------------------------------------'
    logging.info(args_str)

if __name__ == "__main__":
    # 1. args
    args = params.feedback_args()
    log_param(args)

    # 2. check dir and file
    check_dirs_files(dirs=[args.dataset_dir, args.output_dir, ], files=[args.task_file, ])

    model_name = getattr(config, 'MODEL_NAME', None)
    logging.info(f'Using model: {model_name}')

    api_key = config.OPENROUTER_API_KEY or os.getenv('OPENROUTER_API_KEY', '')
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Set it in the environment.")
    with OpenRouter(api_key=api_key) as client:
        feedback(client, args, model_name)