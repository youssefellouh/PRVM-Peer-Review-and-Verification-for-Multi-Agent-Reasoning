import argparse
import os
from datetime import datetime

TASK_FILE = {
    'GSM8K': 'GSM8K.jsonl',         # 1319          
    'ARC-c': 'ARC-c.jsonl',         # 1172      
    'StrategyQA': 'StrategyQA.json',# 2290      
    'MMLU': 'MMLU.json',            # 285
}

EXAMPLE_NUM = {
    'GSM8K': 250,
    'ARC-c': 250,
    'StrategyQA': 250,
    'MMLU': 250,
}
def data_args():
    args_parser = argparse.ArgumentParser(description='process_data')
    # dictionary or file
    args_parser.add_argument('--dataset_dir', type=str, default='datasets')
    args_parser.add_argument('--output_dir', type=str, default='processed_data')

    args_parser.add_argument('--task', type=str, default='GSM8K',
                             choices=['GSM8K','ARC-c','MMLU'])
    args_parser.add_argument('--max_example_num', type=int, default=500)
    args_parser.add_argument('--random_seed', type=int, default=42)
    # parse
    args = args_parser.parse_args()
    args.task_file = os.path.join(os.path.join(args.dataset_dir, args.task), TASK_FILE[args.task])
    args.output_path = os.path.join(args.output_dir, args.task)
    return args

def single_agent_args():
    args_parser = argparse.ArgumentParser(description='single_agent')

    # dictionary or file
    args_parser.add_argument('--dataset_dir', type=str, default='processed_data')
    args_parser.add_argument('--output_dir', type=str, default='result')
    args_parser.add_argument('--task', type=str, default='GSM8K',
                             choices=['GSM8K','ARC-c','MMLU'])
    args_parser.add_argument('--max_example_num', type=int, default=500) # AQuA: 254  AddSub: 395  Colored_Objects:250 Penguins: 146
    args_parser.add_argument('--model', type=str, default=None)
    # agent
    args_parser.add_argument('--agent_num', type=int, default=1) # single agent answer question for 1 time
    # reload data
    args_parser.add_argument('--reload_data', type=bool, default=False)

    # parse
    args = args_parser.parse_args()

    time = datetime.now().strftime("%m%d")
    args.task_file = os.path.join(os.path.join(args.dataset_dir, args.task), f'{args.task}_{args.max_example_num}.jsonl')
    args.output_file = os.path.join(os.path.join(args.output_dir, args.task), f'{args.task}_single_agent_{args.max_example_num}_{time}.json')
    return args

def self_correction():
    args_parser = argparse.ArgumentParser(description='self-correction')
    # dictionary or file
    args_parser.add_argument('--dataset_dir', type=str, default='processed_data')
    args_parser.add_argument('--output_dir', type=str, default='result')

    args_parser.add_argument('--task', type=str, default='GSM8K',
                             choices=['GSM8K','ARC-c','MMLU'])
    args_parser.add_argument('--max_example_num', type=int, default=500) # AQuA: 254  AddSub: 395  Colored_Objects:250   Penguins 146
    # agent
    args_parser.add_argument('--agent_num', type=int, default=1) # single agent for self-critique
    # reload data
    args_parser.add_argument('--reload_data', type=bool, default=False)
    # parse
    args = args_parser.parse_args()
    time = datetime.now().strftime("%m%d")
    args.task_file = os.path.join(os.path.join(args.dataset_dir, args.task), f'{args.task}_{args.max_example_num}.jsonl')
    args.output_file = os.path.join(os.path.join(args.output_dir, args.task), f'{args.task}_self_correction_{args.max_example_num}_{time}.json')
    return args

def self_consistency_args():
    args_parser = argparse.ArgumentParser(description='self-consistency')
    args_parser.add_argument('--dataset_dir', type=str, default='processed_data')
    args_parser.add_argument('--output_dir', type=str, default='result')
    args_parser.add_argument('--task', type=str, default='GSM8K',
                             choices=['GSM8K','ARC-c','MMLU'])
    args_parser.add_argument('--max_example_num', type=int, default=500)
    args_parser.add_argument('--model', type=str, default=None)
    args_parser.add_argument('--sample_num', type=int, default=3)
    args_parser.add_argument('--reload_data', type=bool, default=False)
    args = args_parser.parse_args()
    time = datetime.now().strftime("%m%d")
    args.task_file = os.path.join(os.path.join(args.dataset_dir, args.task), f'{args.task}_{args.max_example_num}.jsonl')
    args.output_file = os.path.join(os.path.join(args.output_dir, args.task), f'{args.task}_self_consistency_{args.max_example_num}_{time}.json')
    return args

def debate_args():
    args_parser = argparse.ArgumentParser(description='debate')

    # dictionary or file
    args_parser.add_argument('--dataset_dir', type=str, default='processed_data')
    args_parser.add_argument('--output_dir', type=str, default='result')

    args_parser.add_argument('--task', type=str, default='GSM8K',
                             choices=['GSM8K','ARC-c','MMLU'])
    args_parser.add_argument('--max_example_num', type=int, default=500) # AQuA: 254  AddSub: 395  Colored_Objects:250
    # agent
    args_parser.add_argument('--agent_num', type=int, default=3)
    args_parser.add_argument('--rounds', type=int, default=3)
    # reload data
    args_parser.add_argument('--reload_data', type=bool, default=False)
    # parse
    args = args_parser.parse_args()
    time = datetime.now().strftime("%m%d")
    args.task_file = os.path.join(os.path.join(args.dataset_dir, args.task), f'{args.task}_{args.max_example_num}.jsonl')
    args.output_file = os.path.join(os.path.join(args.output_dir, args.task), f'{args.task}_debate_{args.max_example_num}_{time}.json')
    return args

def feedback_args():
    args_parser = argparse.ArgumentParser(description='feedback')

    # dictionary or file
    args_parser.add_argument('--dataset_dir', type=str, default='processed_data')
    args_parser.add_argument('--output_dir', type=str, default='result')

    args_parser.add_argument('--task', type=str, default='GSM8K',
                             choices=['GSM8K','ARC-c','MMLU'])
    args_parser.add_argument('--max_example_num', type=int, default=500) # AQuA: 254  AddSub: 395  Colored_Objects:250

    # agent
    args_parser.add_argument('--agent_num', type=int, default=3)
    args_parser.add_argument('--rounds', type=int, default=3) # fix rounds = 3

    # reload data
    args_parser.add_argument('--reload_data', type=bool, default=False)
    # parse
    args = args_parser.parse_args()

    time = datetime.now().strftime("%m%d")
    args.task_file = os.path.join(os.path.join(args.dataset_dir, args.task), f'{args.task}_{args.max_example_num}.jsonl')
    args.output_file = os.path.join(os.path.join(args.output_dir, args.task), f'{args.task}_feedback_{args.max_example_num}_{time}.json')
    return args

def peer_review_args():
    args_parser = argparse.ArgumentParser(description='PRVM_Without_Verifier')
    # dictionary or file
    args_parser.add_argument('--dataset_dir', type=str, default='processed_data')
    args_parser.add_argument('--output_dir', type=str, default='result')
    args_parser.add_argument('--task', type=str, default='GSM8K',
                             choices=['GSM8K','ARC-c','MMLU'])
    args_parser.add_argument('--max_example_num', type=int, default=500) # AQuA: 254  AddSub: 395  Colored_Objects:250
    # agent
    args_parser.add_argument('--agent_num', type=int, default=3)
    args_parser.add_argument('--rounds', type=int, default=4) # fix rounds = 3
    # reload data
    args_parser.add_argument('--reload_data', type=bool, default=False)
    # parse
    args = args_parser.parse_args()
    time = datetime.now().strftime("%m%d")
    args.task_file = os.path.join(os.path.join(args.dataset_dir, args.task), f'{args.task}_{args.max_example_num}.jsonl')
    args.output_file = os.path.join(os.path.join(args.output_dir, args.task), f'{args.task}_peer_review_{args.max_example_num}_{time}.json')
    return args

def PRVM_args():
    args_parser = argparse.ArgumentParser(description='PRVM')
    # dictionary or file
    args_parser.add_argument('--dataset_dir', type=str, default='processed_data')
    args_parser.add_argument('--output_dir', type=str, default='result')
    args_parser.add_argument('--task', type=str, default='GSM8K',
                             choices=['GSM8K','ARC-c','MMLU'])
    args_parser.add_argument('--max_example_num', type=int, default=500) # AQuA: 254  AddSub: 395  Colored_Objects:250
    # agent
    args_parser.add_argument('--agent_num', type=int, default=3)
    args_parser.add_argument('--rounds', type=int, default=4) # fix rounds = 3
    # reload data
    args_parser.add_argument('--reload_data', type=bool, default=False)
    # parse
    args = args_parser.parse_args()
    time = datetime.now().strftime("%m%d")
    args.task_file = os.path.join(os.path.join(args.dataset_dir, args.task), f'{args.task}_{args.max_example_num}.jsonl')
    args.output_file = os.path.join(os.path.join(args.output_dir, args.task), f'{args.task}_PRVM_{args.max_example_num}_{time}.json')
    return args
def eval_args():
    args_parser = argparse.ArgumentParser(description='evaluation')
    # dictionary or file
    args_parser.add_argument('--eval_dir', type=str, default='result')
    args_parser.add_argument('--task', type=str, default='GSM8K',
                             choices=['GSM8K','ARC-c','MMLU'])
    args_parser.add_argument('--method', type=str, default='PRVM_Without_Verifier',
                             choices=['single_agent', 'self_correction', 'self_critique', 'self_consistency','debate', 'feedback', 'PRVM_Without_Verifier', 'PRVM_Without_Confidence', 'PRVM'])
    args_parser.add_argument('--time_flag', type=str, default='1113')
    args_parser.add_argument('--example_num', type=int, default=None)
    # parse
    args = args_parser.parse_args()
    if args.example_num is None:
        args.example_num = EXAMPLE_NUM[args.task]
    if args.method in ['majority', 'single_agent']:
        args.eval_file = os.path.join(os.path.join(args.eval_dir, args.task),
                                      f'{args.task}_single_agent_{args.example_num}_{args.time_flag}.json')
    elif args.method in ['self_critique', 'self_consistency']:
        args.eval_file = os.path.join(os.path.join(args.eval_dir, args.task),
                                      f'{args.task}_{args.method}_{args.example_num}_{args.time_flag}.json')
    else:
        args.eval_file = os.path.join(os.path.join(args.eval_dir, args.task),
                                      f'{args.task}_{args.method}_{args.example_num}_{args.time_flag}.json')
    return args