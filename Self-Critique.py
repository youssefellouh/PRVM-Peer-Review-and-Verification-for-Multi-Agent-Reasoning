
import logging
import os
import time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from data_proc import check_dirs_files
from openrouter import OpenRouter
import json
from tqdm import tqdm
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
            return completion, inference_time
        except Exception as e:
            logging.warning(f"retrying due to an error: {e}")
            time.sleep(20)
def read_jsonl(path: str):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh.readlines() if line]
def self_critique(client, args, model_name):
    """
    Self-Critique Method (3 Rounds):
    Round 0. Generate initial answer
    Round 1. Ask the agent to justify, evaluate, support, and critique its own answer
    Round 2. Generate final improved answer based on self-critique
    """
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
        # -----------------------------ROUND 0: Initial Answer---------------------------------
        if args.task in ['GSM8K']:  # number
            content = """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(question)
        elif args.task in ['ARC-c','MMLU']:  # option
            content = "Can you answer the following question as accurately as possible? {} Explain your answer, putting the answer in the form (X) at the end of your response.".format(question)
        elif args.task in ['StrategyQA']:  # yes or no
            content = "Can you answer the following question as accurately as possible? {} Explain your answer, your answer should be Yes or No at the end of your response.".format(question)
        else:
            raise Exception('failed to construct question, unknown task!')

        agent_contexts = [[{"role": "user", "content": content}] for _ in range(args.agent_num)]  # single agent only
        inference_times = []
        token_usages = []
        for agent_context in agent_contexts:
            completion, inference_time = generate_answer(client, agent_context, model_name)
            inference_times.append(inference_time)
            assistant_message = construct_assistant_message(completion)
            agent_context.append(assistant_message)
            
            # Store token usage
            if hasattr(completion, 'usage') and completion.usage:
                usage = completion.usage
                token_usage = {
                    'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(usage, 'completion_tokens', 0),
                    'total_tokens': getattr(usage, 'total_tokens', 0)
                }
                token_usages.append(token_usage)
            else:
                token_usages.append(None)
        # -----------------------------ROUND 1: Self-Critique---------------------------------
        # Prompt en anglais respectant tes consignes : justifier, évaluer, prouver, et chercher les erreurs.
        critique_content = "Review your previous answer. Please justify your initial response, evaluate its validity, and provide arguments or evidence to support your choices. At the same time, critically analyze your answer to find any potential flaws, errors, or weaknesses."
        
        for _ in range(args.agent_num):
            agent_contexts[_].append({"role": "user", "content": critique_content})

        for agent_context in agent_contexts:
            completion, inference_time = generate_answer(client, agent_context, model_name)
            inference_times.append(inference_time)
            assistant_message = construct_assistant_message(completion)
            agent_context.append(assistant_message)       
            # Store token usage
            if hasattr(completion, 'usage') and completion.usage:
                usage = completion.usage
                token_usage = {
                    'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(usage, 'completion_tokens', 0),
                    'total_tokens': getattr(usage, 'total_tokens', 0)
                }
                token_usages.append(token_usage)
            else:
                token_usages.append(None)
        # -----------------------------ROUND 2: Final Improved Answer---------------------------------
        if args.task in ['GSM8K']:  # number
            final_content = "Based on your self-critique and the evidence you gathered, improve your answer. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."
        elif args.task in ['ARC-c','MMLU']:  # option
            final_content = "Based on your self-critique and the evidence you gathered, improve your answer. You must choose only one option. Please reiterate your answer, with your final answer a single letter, in the form (X)."
        elif args.task in ['StrategyQA']:  # yes or no
            final_content = "Based on your self-critique and the evidence you gathered, improve your answer. Please reiterate your answer, your answer should be Yes or No at the end of your response."
        else:
            final_content = "Based on your self-critique and the evidence you gathered, improve your answer."
        for _ in range(args.agent_num):
            agent_contexts[_].append({"role": "user", "content": final_content})

        for agent_context in agent_contexts:
            completion, inference_time = generate_answer(client, agent_context, model_name)
            inference_times.append(inference_time)
            assistant_message = construct_assistant_message(completion)
            agent_context.append(assistant_message)
            
            # Store token usage
            if hasattr(completion, 'usage') and completion.usage:
                usage = completion.usage
                token_usage = {
                    'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(usage, 'completion_tokens', 0),
                    'total_tokens': getattr(usage, 'total_tokens', 0)
                }
                token_usages.append(token_usage)
            else:
                token_usages.append(None)
        # Sauvegarde
        generated_description.append({
            'question': question,
            'answer': answer,
            'agent_contexts': agent_contexts,
            'inference_times': inference_times,
            'token_usages': token_usages,
        })
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(generated_description, f, ensure_ascii=False)

def log_param(args):
    args_str = f'\n--------------- self critique parameters ---------------\n'
    for k, v in args.__dict__.items():
        args_str += f'{k} = {v}\n'
    args_str += f'-------------------------------------------------------'
    logging.info(args_str)
if __name__ == "__main__":
    # 1. args
    args = params.self_correction()  # Reuse self_correction args structure
    args.output_file = args.output_file.replace('self_correction', 'self_critique')  # distinguish from self_correction.py
    log_param(args)
    # 2. check dir and file
    output_task_dir = os.path.dirname(args.output_file)
    check_dirs_files(dirs=[args.dataset_dir, args.output_dir, output_task_dir], files=[args.task_file, ])
    model_name = getattr(config, 'MODEL_NAME', None)
    logging.info(f'Using model: {model_name}')

    api_key = config.OPENROUTER_API_KEY or os.getenv('OPENROUTER_API_KEY', '')
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Set it in the environment.")
    with OpenRouter(api_key=api_key) as client:
        self_critique(client, args, model_name)