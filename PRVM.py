import logging
import time
import os
import json
import re
from collections import Counter
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
# Supposons que params, config et data_proc sont vos modules locaux
import params
import config
from data_proc import check_dirs_files
from openrouter import OpenRouter
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Silence noisy HTTP/transport libraries to keep output clean
for _name in ['httpx', 'httpcore', 'urllib3', 'asyncio', 'openai', 'stainless', 'groq', 'chardet']:
    logging.getLogger(_name).setLevel(logging.WARNING)

def read_json(input_path):
   with open(input_path, 'r', encoding='utf-8') as f:
       return json.load(f)
def write_json(output_path, output_data):
   with open(output_path, 'w', encoding='utf-8') as f:
       json.dump(output_data, f, ensure_ascii=False)

def read_jsonl(path: str):
   with open(path, encoding='utf-8') as fh:
       return [json.loads(line) for line in fh.readlines() if line]

def construct_assistant_message(completion):
   content = completion.choices[0].message.content
   return {"role": "assistant", "content": content}

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
def generate_answer(client, answer_context, model_name, timeout: int = 30, max_retries: int = 3, backoff: int = 20):
   """Generate an answer with retries and a configurable model."""
   retries = 0
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

       except KeyboardInterrupt:
           logging.info("Interrupted by user during API call.")
           raise
       except Exception as e:
           retries += 1
           logging.warning(f"retrying due to an error: {e}")
           if retries >= max_retries:
               logging.error("Max retries reached due to errors. Raising last exception.")
               raise
           time.sleep(backoff)

# ==========================================
# FONCTIONS DE CONSENSUS MATHEMATIQUE
# ==========================================
def extract_answer(response_text, task_type):
   """Extrait la réponse finale brute selon le type de tâche."""
   if task_type in ['GSM8K']:
       match = re.search(r'\\boxed\{([^}]*)\}', response_text)
       if match:
           return match.group(1).strip()
   elif task_type in ['ARC-c', 'MMLU']:
       match = re.search(r'\(([A-Z])\)', response_text)
       if match:
           return match.group(1).strip()
   elif task_type in ['StrategyQA']:
       text_lower = response_text.lower()
       if text_lower.endswith('yes') or text_lower.endswith('yes.'): return 'Yes'
       if text_lower.endswith('no') or text_lower.endswith('no.'): return 'No'
       match = re.search(r'\b(yes|no)\b', response_text, re.IGNORECASE)
       if match:
           return match.group(1).capitalize()
   return None

def calculate_consensus(answers_list):
   """
   Fonction mathématique : Calcule le Mode (Vote Majoritaire).
   Retourne la réponse consensuelle, le taux de confiance, et le dictionnaire des votes.
   """
   valid_answers = [ans for ans in answers_list if ans is not None]
  
   if not valid_answers:
       return None, 0.0, {}

   counts = Counter(valid_answers)
   most_common_answer, votes = counts.most_common(1)[0]
   confidence = votes / len(answers_list)
  
   return most_common_answer, confidence, dict(counts)
# ==========================================
# AGENT DE VERIFICATION (VERIFIER)
# ==========================================
def create_verifier_prompt():
    return """You are a Senior  Verification Agent in a multi-agent reasoning system.
Your specific role is to strictly evaluate the reasoning process provided by a reasoning agent.

Please evaluate the reasoning based on these criteria:
1. **Logical Consistency**: Check step-by-step progression for logical fallacies.
2. **Mathematical & Factual Accuracy**: Verify all calculations.

If the reasoning is COMPLETELY CORRECT (all steps valid,logically sound):
- Start your response with: [VERIFICATION: VALID]
- Briefly explain why it is correct.

If you find ANY error (logical or calculation), or if the final answer is incorrect:
- Start your response with: [VERIFICATION: INVALID]
- **Error Found:** [Quote the specific error or mistake]
- **Why It's Wrong:** [Detailed explanation of the error]

Be strict and thorough. The agent depends on your feedback to improve."""

def verify_reasoning(client, reasoning, question, expected_answer=None, model_name=None):
   verifier_prompt = create_verifier_prompt()
   verify_prompt = f"{verifier_prompt}\n\nQuestion: {question}\n\nReasoning to verify:\n{reasoning}\n"
   if expected_answer is not None:
       verify_prompt += f"\nGroup consensus / expected answer: {expected_answer}\n"
   verify_prompt += "\nYour verification:"

   context = [{"role": "user", "content": verify_prompt}]
   completion, inference_time = generate_answer(client, context, model_name)
   if completion is None:
       logging.error("Verification failed: No response from API")
       return False, "Error: No response from verification API", 0

   verification_output = completion.choices[0].message.content
   # Be robust: consider it valid only if the response starts with the valid tag
   is_valid = isinstance(verification_output, str) and verification_output.strip().startswith('[VERIFICATION: VALID]')

   return is_valid, verification_output, inference_time
# ==========================================
# BOUCLE PRINCIPALE : PEER REVIEW + CONSENSUS + VERIFICATION
# ==========================================
def peer_review(client, args, model_name):
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
    MAX_ATTEMPTS = 2 # Limite de tentatives si les réponses sont invalides


    iterable = tqdm(all_datas) if not getattr(args, 'no_progress', False) else all_datas
    for i, data in enumerate(iterable):

       if args.reload_data and i < generated_len:
           continue
       question = data['question']
       answer = data['answer']
       # Construction du prompt selon la tâche
       if args.task in ['GSM8K']:
           base_content = f"Can you solve the following math problem? {question} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."
       elif args.task in ['ARC-c','MMLU']:
           base_content = f"Can you answer the following question as accurately as possible? {question} Explain your answer, putting the answer in the form (X) at the end of your response."
       elif args.task in ['StrategyQA']:
           base_content = f"Can you answer the following question as accurately as possible? {question} Explain your answer, your answer should be Yes or No at the end of your response."
       else:
           raise Exception('failed to construct question, unknown task!')
       previous_error_feedback = ""
       total_inference_times = []
       is_valid = False
       attempt = 0
       consensus_ans = None
       confidence = 0.0
       votes_dict = {}
       # Boucle de tentatives
       for attempt in range(MAX_ATTEMPTS):
        
           current_content = base_content
           if previous_error_feedback:
               current_content += f"\n\n[System Note]: In a previous attempt, the system rejected the answer. Feedback:\n{previous_error_feedback}\n\nPlease correct these mistakes and try again."


           agent_contexts = [[{"role": "user", "content": current_content}] for _ in range(args.agent_num)]
           agent_init_ans = None
           agent_feedbacks = [[] for _ in range(args.agent_num)]
           token_usages = []
          
           # Boucle des rounds de discussion entre agents
           for round_num in range(args.rounds):
               if round_num == 1:
                   agent_init_ans = [agent_contexts[k][1]['content'] for k in range(args.agent_num)]

               for j, agent_context in enumerate(agent_contexts):
                   if round_num == 0:
                       completion, inference_time = generate_answer(client, agent_context, model_name)
                       total_inference_times.append(inference_time)
                       token_usages.append(get_token_usage(completion))
                       assistant_message = construct_assistant_message(completion)
                       agent_context.append(assistant_message)
                      
                   elif round_num == 1:
                       ans_to_add = [k for k in range(args.agent_num) if k != j]
                       for index in ans_to_add:
                           init_ans = agent_init_ans[index]
                           content = f"Here is a solution from another agent: \n\n {init_ans}\n\n Please examine this agent's reasoning process step by step and offer feedback on its reasoning. You can rate your confidence in your feedback on a scale from 1-10."
                           agent_context.append({"role": "user", "content": content})
                           completion, inference_time = generate_answer(client, agent_context, model_name)
                           total_inference_times.append(inference_time)
                           token_usages.append(get_token_usage(completion))
                           assistant_message = construct_assistant_message(completion)
                           agent_context.append(assistant_message)
                           agent_feedbacks[index].append(assistant_message)
                          
                   elif round_num == 2:
                       agent_feedback = agent_feedbacks[j]
                       agent_num_dict = {1: "one", 2: "two", 3: "three", 4: "four"}
                       content = f"Here are the feedbacks for your solution from the above {agent_num_dict.get(args.agent_num - 1, str(args.agent_num - 1))} agents:\n\n "
                       for feedback in agent_feedback:
                           content += f"One agent feedback: {feedback['content']} \n\n "

                       if args.task in ['GSM8K']:
                           content += f"Using other agents' solutions and feedbacks as additional information, can you provide your answer to the math problem? \n The original math problem is {question}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."
                       elif args.task in ['ARC-c','MMLU']:
                           content += f"Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and other agents' feedback step by step. Put your answer in the form (X) at the end of your response."
                       elif args.task in ['StrategyQA']:
                           content += f"Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and other agents' feedback step by step. Your answer should be Yes or No at the end of your response."

                       agent_context.append({"role": "user", "content": content})
                       completion, inference_time = generate_answer(client, agent_context, model_name)
                       total_inference_times.append(inference_time)
                       token_usages.append(get_token_usage(completion))
                       assistant_message = construct_assistant_message(completion)
                       agent_context.append(assistant_message)

           # --------------------------------------------------------
           # ETAPE 2 : CALCUL DU CONSENSUS MATHEMATIQUE ET VERIFICATION
           # --------------------------------------------------------          
           # A) Extraction de toutes les réponses finales
           agent_final_answers = []
           for j in range(args.agent_num):
               agent_text = agent_contexts[j][-1]['content']
               ans = extract_answer(agent_text, args.task)
               agent_final_answers.append(ans)
          
           # B) Calcul mathématique du consensus (Mode)
           consensus_ans, confidence, votes_dict = calculate_consensus(agent_final_answers)
           logging.info(f"Question {i} (Tentative {attempt + 1}) - Votes: {votes_dict} -> Consensus: {consensus_ans} (Confiance: {confidence*100:.1f}%)")

           # C) Sélection du raisonnement de l'agent correspondant au consensus
           final_reasoning = ""
           for j in range(args.agent_num):
               if agent_final_answers[j] == consensus_ans:
                   final_reasoning = agent_contexts[j][-1]['content']
                   break
          
           # Sécurité si aucun format n'a été respecté
           if not final_reasoning:
               final_reasoning = agent_contexts[0][-1]['content']
           # D) Envoi de la réponse consensuelle au Vérificateur (Agent indépendant)
           # Auto-accept: si la confiance > 50%, on accepte directement la réponse finale.
           if confidence > 0.5:
               logging.info(f"Question {i} : Auto-accept consensus sans vérification (Tentative {attempt + 1}). Confidence={confidence*100:.1f}%")
               is_valid = True
               verification_feedback = "Auto-accepted: confidence > 50%"
               verif_time = 0.0
           else:
               is_valid, verification_feedback, verif_time = verify_reasoning(client, final_reasoning, question, consensus_ans, model_name)

           total_inference_times.append(verif_time)

           if is_valid:
               logging.info(f"Question {i} : Verification VALIDE (Tentative {attempt + 1})")
               break # Succès : On sort de la boucle des tentatives
           else:
               logging.warning(f"Question {i} : Verification INVALIDE (Tentative {attempt + 1}). Recommence...")
               # Injection du vote et de l'erreur pour aider lors de la tentative suivante
               previous_error_feedback = f"Your previous group consensus was '{consensus_ans}' with votes {votes_dict}, but it failed verification. The verifier found this error:\n{verification_feedback}"
              
       # --------------------------------------------------------
       # ETAPE 3 : SAUVEGARDE DES RESULTATS (INCLUANT LE CONSENSUS)
       # --------------------------------------------------------
       generated_description.append({
           'question': question,
           'ground_truth': answer,
           'consensus_answer': consensus_ans,
           'confidence': confidence,
           'votes_dict': votes_dict,
           'agent_contexts': agent_contexts,
           'inference_times': total_inference_times,
           'token_usages': token_usages,
           'attempts_used': attempt + 1,
           'is_valid_final': is_valid,
           'verification_feedback': verification_feedback
       })
      
       with open(args.output_file, 'w', encoding='utf-8') as f:
           json.dump(generated_description, f, ensure_ascii=False, indent=4)
    # --------------------------------------------------------
    # AFFICHAGE DES METRIQUES GLOBALES A LA FIN
    # --------------------------------------------------------
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

        print("\n" + "="*50)
        print("--- Metriques de Generation Multi-Agents ---")

        if total_tokens_list:
            token_mean = np.mean(total_tokens_list)
            token_total = sum(total_tokens_list)
            print(f"Token Usage: mean={token_mean:.1f} tokens/call, total={token_total} tokens")

        if all_inference_times:
            time_mean = np.mean(all_inference_times)
            time_std = np.std(all_inference_times)
            time_count = len(all_inference_times)
            print(f"Inference time: mean={time_mean:.4f}s, std={time_std:.4f}s, calls={time_count}")
        print("="*50 + "\n")

def log_param(args):
   args_str = f'\n---------------- PRVM parameters ----------------\n'
   for k, v in args.__dict__.items():
       args_str += f'{k} = {v}\n'
   args_str += f'-------------------------------------------------------'
   logging.info(args_str)
if __name__ == "__main__":
    # 1. Initialisation des arguments
    args = params.PRVM_args()
    log_param(args)

    # 2. Verification des dossiers et fichiers
    output_task_dir = os.path.dirname(args.output_file)
    check_dirs_files(dirs=[args.dataset_dir, args.output_dir, output_task_dir], files=[args.task_file, ])
  
    # 3. Configuration de l'API OpenRouter
    model_name = getattr(config, 'MODEL_NAME', None)
    logging.info(f'Using model: {model_name}')
    client = OpenRouter(api_key=config.OPENROUTER_API_KEY)
    # 4. Lancement de la procédure
    try:
        peer_review(client, args, model_name)
    except KeyboardInterrupt:
        logging.warning("Execution interrupted by user (KeyboardInterrupt). Exiting gracefully.")







