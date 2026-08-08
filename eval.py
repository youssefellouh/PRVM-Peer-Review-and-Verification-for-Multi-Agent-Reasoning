
import logging
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up metrics logger
metrics_logger = logging.getLogger('metrics')
metrics_logger.setLevel(logging.INFO)
metrics_logger.propagate = False
metrics_handler = logging.FileHandler('evaluation_metrics.log')
metrics_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
metrics_logger.addHandler(metrics_handler)

from data_proc import check_dirs_files
import glob
import json
import jsonlines
import numpy as np
import re
from tqdm import tqdm
from sklearn.metrics import roc_curve, precision_recall_fscore_support
from scipy.interpolate import interp1d
import time
from typing import List, Tuple, Any, Dict
from collections import Counter
from params import eval_args
def read_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)
def read_jsonl(input_path):
    output_data = []
    with open(input_path, 'r+', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            output_data.append(item)
    return output_data

def delete_extra_zero(n):
    try:
        n = float(n)
    except:
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)
        n = str(n)
        return n
def parse_gt_answer(input_str, args):
    a = None
    if args.task == 'GSM8K': 
        a = delete_extra_zero(input_str.split("#### ")[-1].replace(",", ""))
    elif args.task in ['ARC-c', 'StrategyQA','MMLU']: 
        a = input_str
    else:
        raise Exception('failed to parse the answer, unknown task!')
    assert a
    return a

def parse_pred_answer(input_str, args):
    solution = None
    if args.task in ['GSM8K']: 
        pattern = r"\{([-0-9.,$]*)\}" 
        matches = re.findall(pattern, input_str)
        if not matches: 
            matches = re.findall(r'-?\d+(?:\.\d+)?', input_str)

        for match_str in matches[::-1]:
            solution = re.sub(r"[^-0-9.]", "", match_str)
            if solution:
                break
    elif args.task in ['ARC-c','MMLU']: 
        pattern = r'\b([A-E])\b|\(([A-E])\)' 
        matches = re.findall(pattern, input_str)
        for match_tuple in matches[::-1]:
            solution = next((m for m in match_tuple if m), None)
            if solution and solution.upper() != 'X':
                solution = solution.upper()
                break
    elif args.task in ['StrategyQA', ]: 
        pattern = r'\b(YES|Yes|yes|NO|No|no)\b'
        matches = re.findall(pattern, input_str)
        for match_str in matches[::-1]:
            if match_str in ['Yes', 'yes', 'YES']:
                solution = 'Yes'
            if match_str in ['No', 'no', 'NO']:
                solution = 'No'
            if solution:
                break
    return solution

def most_frequent(lst):
    if not lst:
        return None
    return Counter(lst).most_common(1)[0][0]

def compute_accuracy(gt, pred_solutions, args):
    gt_answer = parse_gt_answer(gt, args)
    if gt_answer is None:
        raise Exception('could not parse ground truth answer!')

    if type(pred_solutions) == list: 
        pred_answers = []
        for pred_solution in pred_solutions:
            pred_answer = parse_pred_answer(pred_solution, args)
            if pred_answer: 
                pred_answers.append(pred_answer)
        pred_answer = most_frequent(pred_answers)
    else: 
        pred_answer = parse_pred_answer(pred_solutions, args)

    if pred_answer is None: 
        return 0

    if args.task in ['GSM8K']:
        try:
            if float(gt_answer) == float(pred_answer): 
                return 1
            else:
                return 0
        except:
            return 0
    elif args.task in ['ARC-c', 'StrategyQA','MMLU']: 
        if gt_answer == pred_answer:
            return 1
        else:
            return 0
    else:
        raise Exception('failed to parse the answer, unknown task!')
def compute_precision_recall_f1(gt_list, pred_list, task=None):
    filtered = [(gt, pred) for gt, pred in zip(gt_list, pred_list) if pred is not None]
    if not filtered:
        return None, None, None

    y_true, y_pred = zip(*filtered)
    try:
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0)
        return precision * 100, recall * 100, f1_score * 100
    except Exception:
        return None, None, None

def aggregate_inference_times(input_data):
    times = []
    for tmp_data in input_data:
        if 'inference_times' in tmp_data and isinstance(tmp_data['inference_times'], list):
            for t in tmp_data['inference_times']:
                try:
                    times.append(float(t))
                except Exception:
                    continue
    if not times:
        return None, None, 0
    return float(np.mean(times)), float(np.std(times)), len(times)

# ==============================================================
# NOUVELLES FONCTIONS POUR TOKEN USAGE ET RECTIFICATION RATE
# ==============================================================
def aggregate_token_usages(input_data):
    """Calcule la moyenne et le total des tokens utilisés depuis les données JSON."""
    total_tokens_list = []
    for tmp_data in input_data:
        if 'total_tokens' in tmp_data:
            total_tokens_list.append(int(tmp_data['total_tokens']))
        elif 'usage' in tmp_data and isinstance(tmp_data['usage'], dict) and 'total_tokens' in tmp_data['usage']:
            total_tokens_list.append(int(tmp_data['usage']['total_tokens']))
        elif 'token_usages' in tmp_data and tmp_data['token_usages']:
            # For single_agent format: list of dicts with total_tokens
            for usage in tmp_data['token_usages']:
                if usage and 'total_tokens' in usage:
                    total_tokens_list.append(int(usage['total_tokens']))
        else:
            sum_tokens = 0
            found_tokens = False
            for response in tmp_data.get('agent_contexts', []):
                for msg in response:
                    if 'total_tokens' in msg:
                        sum_tokens += int(msg['total_tokens'])
                        found_tokens = True
                    elif 'usage' in msg and 'total_tokens' in msg['usage']:
                        sum_tokens += int(msg['usage']['total_tokens'])
                        found_tokens = True
            if found_tokens:
                total_tokens_list.append(sum_tokens)

    if not total_tokens_list:
        return None, None
    return float(np.mean(total_tokens_list)), sum(total_tokens_list)

def get_initial_solution(response):
    """Extrait la PREMIÈRE réponse (draft initial) d'un agent pour calculer la rectification."""
    for msg in response:
        if msg.get('role') in ['assistant', 'model']:
            return msg.get('content', '')
    if len(response) > 1:
        return response[1].get('content', '')
    elif len(response) > 0:
        return response[0].get('content', '')
    return ""
# ==============================================================
def count_prediction_validity(pred_list):
    valid = sum(1 for pred in pred_list if pred is not None)
    invalid = sum(1 for pred in pred_list if pred is None)
    return valid, invalid

def log_param(args):
    args_str = f'\n--------------- evaluation parameters -----------------\n'
    for k, v in args.__dict__.items():
        args_str += f'{k} = {v}\n'
    args_str += f'-------------------------------------------------------'
    logging.info(args_str)


def resolve_eval_file(args):
    expected_path = os.path.join(
        args.eval_dir,
        args.task,
        f'{args.task}_{args.method}_{args.example_num}_{args.time_flag}.json'
    )
    if os.path.exists(expected_path):
        return expected_path

    if args.method in ['majority', 'single_agent']:
        pattern = os.path.join(args.eval_dir, args.task, f'{args.task}_single_agent_{args.example_num}_*.json')
    elif args.method in ['self_critique', 'self_consistency']:
        pattern = os.path.join(args.eval_dir, args.task, f'{args.task}_{args.method}_{args.example_num}_*.json')
    else:
        pattern = os.path.join(args.eval_dir, args.task, f'{args.task}_{args.method}_{args.example_num}_*.json')

    candidates = sorted(glob.glob(pattern))
    if not candidates and args.method == 'PRVM_Without_Confidence':
        fallback_pattern = os.path.join(args.eval_dir, args.task, f'*PRVM_Without_Confidence*{args.example_num}*.json')
        candidates = sorted(glob.glob(fallback_pattern))

    if candidates:
        selected = max(candidates, key=os.path.getmtime)
        logging.info(f"Using evaluation file: {selected}")
        return selected

    logging.warning(f"No evaluation file found for pattern: {pattern}")
    return expected_path

def evaluate_multi_agent(args):
    input_data = read_json(args.eval_file)
    if len(input_data) != args.example_num:
        logging.warning(f"Number of entries in file ({len(input_data)}) does not match expected ({args.example_num}). Evaluating on available data.")
    accuracies = []
    # Nouveaux compteurs pour la Rectification
    initially_incorrect_count = 0
    rectified_count = 0
    index = 0
    for tmp_data in tqdm(input_data):
        index += 1
        responses = tmp_data['agent_contexts']
        gt = tmp_data.get('answer', tmp_data.get('ground_truth'))

        # 1. Extraction Finale
        pred_solutions = []
        for response in responses:
            pred_solution = response[-1]['content']
            pred_solutions.append(pred_solution)
        accurate = compute_accuracy(gt, pred_solutions, args)
        
        # 2. Extraction Initiale (Pour Rectification Rate)
        initial_solutions = [get_initial_solution(response) for response in responses]
        initial_accurate = compute_accuracy(gt, initial_solutions, args)

        # Calcul de la rectification
        if not initial_accurate:
            initially_incorrect_count += 1
            if accurate:
                rectified_count += 1

        tmp_data['result'] = accurate
        if accurate:
            accuracies.append(float(accurate))
        else:
            accuracies.append(0.0)

        if index % 100 == 0 or index == args.example_num:
            acc = np.mean(accuracies) * 100
            sem = np.std(accuracies) / (len(accuracies) ** 0.5) * 100
            metrics_logger.info(f"{index} accuracy: {acc:.4f} %, SEM: {sem:.4f} %")

    # Final result
    final_acc = np.mean(accuracies) * 100
    final_sem = np.std(accuracies) / (len(accuracies) ** 0.5) * 100 if len(accuracies) > 1 else 0

    # Taux de Rectification (Calcul)
    rectification_rate = (rectified_count / initially_incorrect_count * 100) if initially_incorrect_count > 0 else 0.0

    gt_list = [parse_gt_answer(tmp_data.get('answer', tmp_data.get('ground_truth')), args) for tmp_data in input_data]
    pred_list = []
    for tmp_data in input_data:
        responses = tmp_data['agent_contexts']
        pred_solutions = [response[-1]['content'] for response in responses]
        pred_answer = most_frequent([parse_pred_answer(pred, args) for pred in pred_solutions if parse_pred_answer(pred, args)])
        pred_list.append(pred_answer)
    precision, recall, f1_score = compute_precision_recall_f1(gt_list, pred_list, args.task)
    valid_preds, invalid_preds = count_prediction_validity(pred_list)

    print(f"Accuracy: {final_acc:.4f} %, SEM: {final_sem:.4f} %")
    print(f"Valid predictions: {valid_preds}, Invalid predictions: {invalid_preds}")
    if precision is not None:
        print(f"Precision: {precision:.4f} %, Recall: {recall:.4f} %, F1-Score: {f1_score:.4f} %")

    time_mean, time_std, time_count = aggregate_inference_times(input_data)
    if time_mean is not None:
        print(f"Inference time: mean={time_mean:.4f}s, std={time_std:.4f}s, calls={time_count}")

    # ================= LOGS DES NOUVELLES METRIQUES =================
    token_mean, token_total = aggregate_token_usages(input_data)
    # Log comprehensive metrics to file as in single-agent evaluation
    log_comprehensive_metrics(
        args,
        final_acc,
        final_sem,
        valid_preds,
        invalid_preds,
        precision,
        recall,
        f1_score,
        time_mean,
        time_std,
        time_count,
        rectification_rate,
        rectified_count,
        initially_incorrect_count,
        token_mean,
        token_total,
    )
    # ================================================================
def evaluate_self_consistency(args):
    input_data = read_json(args.eval_file)
    if len(input_data) != args.example_num:
        logging.warning(f"Number of entries in file ({len(input_data)}) does not match expected ({args.example_num}). Evaluating on available data.")
    accuracies = []
    gt_lst, pred_solutions_lst = [], []

    for tmp_data in tqdm(input_data):
        gt = tmp_data.get('answer', tmp_data.get('ground_truth'))
        gt_lst.append(gt)

        sampled_answers = tmp_data.get('sampled_answers', [])
        if sampled_answers:
            pred_solutions_lst.append(sampled_answers)
        else:
            responses = tmp_data.get('agent_contexts', [])
            pred_solutions_lst.append([response[-1]['content'] for response in responses if response and len(response) > 0])

    for gt, pred_solutions in zip(gt_lst, pred_solutions_lst):
        accurate = compute_accuracy(gt, pred_solutions, args)
        accuracies.append(float(accurate))

    final_acc = np.mean(accuracies) * 100 if accuracies else 0.0
    final_sem = np.std(accuracies) / (len(accuracies) ** 0.5) * 100 if len(accuracies) > 1 else 0.0
    all_pred = []
    for gt, pred_solutions in zip(gt_lst, pred_solutions_lst):
        parsed_answers = [parse_pred_answer(pred, args) for pred in pred_solutions if parse_pred_answer(pred, args)]
        all_pred.append(most_frequent(parsed_answers))

    precision, recall, f1_score = compute_precision_recall_f1(gt_lst, all_pred, args.task)
    valid_preds, invalid_preds = count_prediction_validity(all_pred)

    print(f"Accuracy: {final_acc:.4f} %, SEM: {final_sem:.4f} %")
    print(f"Valid predictions: {valid_preds}, Invalid predictions: {invalid_preds}")
    if precision is not None:
        print(f"Precision: {precision:.4f} %, Recall: {recall:.4f} %, F1-Score: {f1_score:.4f} %")
    time_mean, time_std, time_count = aggregate_inference_times(input_data)
    if time_mean is not None:
        print(f"Inference time: mean={time_mean:.4f}s, std={time_std:.4f}s, calls={time_count}")

    token_mean, token_total = aggregate_token_usages(input_data)
    log_comprehensive_metrics(
        args,
        final_acc,
        final_sem,
        valid_preds,
        invalid_preds,
        precision,
        recall,
        f1_score,
        time_mean,
        time_std,
        time_count,
        0.0,
        0,
        0,
        token_mean,
        token_total,
    )
def evaluate_single_agent(args):
    input_data = read_json(args.eval_file)
    if len(input_data) != args.example_num:
        logging.warning(f"Number of entries in file ({len(input_data)}) does not match expected ({args.example_num}). Evaluating on available data.")

    gt_lst, pred_solutions_lst, initial_solutions_lst = [], [], []
    
    for tmp_data in tqdm(input_data):
        responses = tmp_data['agent_contexts']
        gt = tmp_data.get('answer', tmp_data.get('ground_truth'))
        gt_lst.append(gt)
        pred_solutions = []
        initial_solutions = []
        for response in responses:
            pred_solutions.append(response[-1]['content'])
            initial_solutions.append(get_initial_solution(response))
        pred_solutions_lst.append(pred_solutions)
        initial_solutions_lst.append(initial_solutions)
    single_agent_num = len(pred_solutions_lst[0])
    multi_agent_result = [{} for i in range(single_agent_num)]    
    # Statistiques globales de rectification pour tous les agents
    total_initially_incorrect = 0
    total_rectified = 0

    for agent_num in range(single_agent_num):
        accuracies = []
        agent_pred_solutions = [item[agent_num] for item in pred_solutions_lst]
        agent_init_solutions = [item[agent_num] for item in initial_solutions_lst]        
        agent_init_incorrect = 0
        agent_rectified = 0

        index = 0
        for gt, pred_solution, init_solution in zip(gt_lst, agent_pred_solutions, agent_init_solutions):
            index += 1
            accurate = compute_accuracy(gt, pred_solution, args)
            init_accurate = compute_accuracy(gt, init_solution, args)
            
            # Rectification rate check
            if not init_accurate:
                agent_init_incorrect += 1
                if accurate:
                    agent_rectified += 1

            if accurate:
                accuracies.append(float(accurate))
            else:
                accuracies.append(0.0)

            if index % 100 == 0 or index == args.example_num:
                acc, sem = np.mean(accuracies) * 100, np.std(accuracies) / (len(accuracies) ** 0.5) * 100
                metrics_logger.info(f"{index} accuracy: {acc:.4f} %, SEM: {sem:.4f} %")
                multi_agent_result[agent_num][index] = (acc, sem)

        total_initially_incorrect += agent_init_incorrect
        total_rectified += agent_rectified

        final_acc = np.mean(accuracies) * 100
        final_sem = np.std(accuracies) / (len(accuracies) ** 0.5) * 100 if len(accuracies) > 1 else 0
        multi_agent_result[agent_num][len(input_data)] = (final_acc, final_sem)

    final_key = max(multi_agent_result[0].keys())
    final_mean_acc = sum([multi_agent_result[i][final_key][0] for i in range(single_agent_num)]) / single_agent_num
    final_mean_sem = sum([multi_agent_result[i][final_key][1] for i in range(single_agent_num)]) / single_agent_num

    all_gt = []
    all_pred = []
    for agent_num in range(single_agent_num):
        agent_pred_solutions = [item[agent_num] for item in pred_solutions_lst]
        for gt, pred in zip(gt_lst, agent_pred_solutions):
            all_gt.append(parse_gt_answer(gt, args))
            all_pred.append(parse_pred_answer(pred, args) if pred else None)

    precision, recall, f1_score = compute_precision_recall_f1(all_gt, all_pred, args.task)
    valid_preds, invalid_preds = count_prediction_validity(all_pred)

    print(f"Accuracy: {final_mean_acc:.4f} %, SEM: {final_mean_sem:.4f} %")
    print(f"Valid predictions: {valid_preds}, Invalid predictions: {invalid_preds}")
    if precision is not None:
        print(f"Precision: {precision:.4f} %, Recall: {recall:.4f} %, F1-Score: {f1_score:.4f} %")

    time_mean, time_std, time_count = aggregate_inference_times(input_data)
    if time_mean is not None:
        print(f"Inference time: mean={time_mean:.4f}s, std={time_std:.4f}s, calls={time_count}")

    # ================= LOGS DES NOUVELLES METRIQUES =================
    rectification_rate = (total_rectified / total_initially_incorrect * 100) if total_initially_incorrect > 0 else 0.0
    token_mean, token_total = aggregate_token_usages(input_data)
    
    # Log comprehensive metrics
    log_comprehensive_metrics(args, final_mean_acc, final_mean_sem, valid_preds, invalid_preds, precision, recall, f1_score, time_mean, time_std, time_count, rectification_rate, total_rectified, total_initially_incorrect, token_mean, token_total)
    # ================================================================
# [Reste du fichier original (calculate_tar_at_far, calculate_alignment_metrics, etc.) inchangé]

def calculate_tar_at_far(y_true, scores, target_far=0.001):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    tar_interpolator = interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value=0.0)
    tar_at_far = tar_interpolator(target_far)
    return tar_at_far

def calculate_alignment_metrics(c_gen: List[Any], c_real: List[Any], alignments: List[Tuple[Any, Any]]) -> Dict[str, float]:
    if len(c_gen) == 0 or len(c_real) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    matched_gen_elements = set(gen_item for gen_item, real_item in alignments)
    matched_real_elements = set(real_item for gen_item, real_item in alignments)
    precision = len(matched_gen_elements) / len(c_gen)
    recall = len(matched_real_elements) / len(c_real)

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
def measure_inference_time(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    return result, inference_time
def log_comprehensive_metrics(args, accuracy, sem, valid_preds, invalid_preds, precision, recall, f1_score, time_mean, time_std, time_count, rectification_rate, rectified_count, initially_incorrect_count, token_mean, token_total):
    """Log one concise evaluation summary to the console and the metrics file."""
    summary_lines = [
        "=" * 80,
        f"----------- evaluation parameters {args.method} -----------",
        f"eval_dir = {args.eval_dir}",
        f"task = {args.task}",
        f"method = {args.method}",
        f"time_flag = {args.time_flag}",
        f"example_num = {args.example_num}",
        f"eval_file = {args.eval_file}",
        "-" * 80,
        "Metrics:",
        f"Accuracy: {accuracy:.4f} %, SEM: {sem:.4f} %",
        f"Valid predictions: {valid_preds}, Invalid predictions: {invalid_preds}",
    ]
    if precision is not None:
        summary_lines.append(f"Precision: {precision:.4f} %, Recall: {recall:.4f} %, F1-Score: {f1_score:.4f} %")
    if time_mean is not None:
        summary_lines.append(f"Inference time: mean={time_mean:.4f}s, std={time_std:.4f}s, calls={time_count}")
    summary_lines.append(f"Global Rectification Rate: {rectification_rate:.2f} % ({rectified_count}/{initially_incorrect_count} initially wrong corrected)")
    if token_mean is not None:
        summary_lines.append(f"Global Token Usage: mean={token_mean:.1f} tokens/problem, total={token_total} tokens")
    summary_lines.append("=" * 80)

    for line in summary_lines:
        logging.info(line)
        metrics_logger.info(line)

if __name__ == "__main__":
    # 1. args
    args = eval_args()
    log_param(args)
    args.eval_file = resolve_eval_file(args)
    # 2. check dir and file
    check_dirs_files(dirs=[args.eval_dir, ], files=[args.eval_file, ])
    # 3. evaluation
    if args.method == 'self_consistency':
        evaluate_self_consistency(args)
    elif args.method in ['single_agent', 'self_correction', 'self_critique']:
        evaluate_single_agent(args)
    else:
        evaluate_multi_agent(args)