import logging
import os
import time
from datetime import datetime
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from data_proc import check_dirs_files
import json
from tqdm import tqdm
import params
from openrouter import OpenRouter
import config

def read_json(input_path):
   with open(input_path, 'r', encoding='utf-8') as f:
       return json.load(f)

def write_json(output_path, output_data):
   with open(output_path, 'w', encoding='utf-8') as f:
       json.dump(output_data, f, ensure_ascii=False)
def construct_assistant_message(completion):
   if isinstance(completion, dict):
       content = completion["choices"][0]["message"]["content"]
   else:
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
            time.sleep(0.5)
            return completion, inference_time
       except Exception as e:
           logging.warning(f"retrying due to an error: {e}")
           time.sleep(20)
def read_jsonl(path: str):
   with open(path, encoding='utf-8') as fh:
       return [json.loads(line) for line in fh.readlines() if line]

def peer_review(args, model_name):
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
       elif args.task in ['StrategyQA', ]:  # yes or no
           content = "Can you answer the following question as accurately as possible? {} Explain your answer, your answer should be Yes or No at the end of your response.".format(question)
       else:
           raise Exception('failed to construct question, unknown task!')
       agent_contexts = [[{"role": "user", "content": content}] for _ in range(args.agent_num)]

       agent_init_ans = None # agent initial answer
       agent_feedbacks = [[] for _ in range(args.agent_num)] # agent feedback
       for round_num in range(args.rounds):

           if  round_num == 1: # update agent initial answer
               agent_init_ans = [agent_contexts[k][1]['content'] for k in range(args.agent_num)]

           for j, agent_context in enumerate(agent_contexts):
               if round_num == 0: # ROUND 0: generate initial answer
                   completion, _ = generate_answer(client, agent_context, model_name)
                   assistant_message = construct_assistant_message(completion)
                   agent_context.append(assistant_message)
               elif round_num == 1: # ROUND 1: give feedback to each other
                   ans_to_add = [k for k in range(args.agent_num) if k != j]
                   for index in ans_to_add:
                       init_ans = agent_init_ans[index]
                       content = f"Here is a solution from another agent: \n\n {init_ans}\n\n Please examine this agent's reasoning process step by step and offer feedback on its reasoning. You can rate your confidence in your feedback on a scale from 1-10."
                       agent_context.append({"role": "user", "content": content})
                       completion, _ = generate_answer(client, agent_context, model_name)
                       assistant_message = construct_assistant_message(completion)
                       agent_context.append(assistant_message)
                       agent_feedbacks[index].append(assistant_message)
               elif round_num == 2: # ROUND 2: base on the initial answer and other angents' feedback, update final answer
                   agent_feedback = agent_feedbacks[j]
                   agent_num_dict = {1: "one", 2: "two", 3: "three", 4: "four"}
                   content = f"Here are the feedbacks for your solution from the above {agent_num_dict.get(args.agent_num - 1, args.agent_num - 1)} agents:\n\n "
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
                   completion, _ = generate_answer(client, agent_context, model_name)
                   assistant_message = construct_assistant_message(completion)
                   agent_context.append(assistant_message)

       generated_description.append({
           'question': question,
           'answer': answer,
           'agent_contexts': agent_contexts,
       })
       with open(args.output_file, 'w', encoding='utf-8') as f:
           json.dump(generated_description, f, ensure_ascii=False)

def log_param(args):
   args_str = f'\n---------------- peer review parameters ----------------\n'
   for k, v in args.__dict__.items():
       args_str += f'{k} = {v}\n'
   args_str += f'-------------------------------------------------------'
   logging.info(args_str)

if __name__ == "__main__":
   args = params.peer_review_args()
   time_flag = datetime.now().strftime("%m%d")
   args.output_file = os.path.join(args.output_dir, args.task,
                                   f'{args.task}_PRVM_Without_Verifier_{args.max_example_num}_{time_flag}.json')
   log_param(args)
   check_dirs_files(dirs=[args.dataset_dir, args.output_dir, os.path.dirname(args.output_file)], files=[args.task_file, ])

   model_name = getattr(config, 'MODEL_NAME', None) 
   logging.info(f'Using model: {model_name}')
   api_key = config.OPENROUTER_API_KEY or os.getenv('OPENROUTER_API_KEY', '')
   if not api_key:
       raise RuntimeError("OPENROUTER_API_KEY is missing. Set it in the environment.")
   with OpenRouter(api_key=api_key) as client:
       peer_review(args, model_name)

