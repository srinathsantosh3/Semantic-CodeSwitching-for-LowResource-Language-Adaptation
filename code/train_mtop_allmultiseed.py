import torch
import torch.nn as nn
import numpy as np
import random
import os
import gc
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import concatenate_datasets
from adapters import AutoAdapterModel, SeqBnConfig, Stack
from nltk import word_tokenize
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    DataCollatorWithPadding, 
    get_linear_schedule_with_warmup 
)
from torch.optim import AdamW 
from torch.utils.data import DataLoader 
from peft import get_peft_model, LoraConfig, TaskType
import nltk
# --- 1. Global Configuration ---
BASE_MODEL = "xlm-roberta-base" #TODO
#BASE_MODEL = "sentence-transformers/LaBSE"
#BASE_MODEL = "bert-base-multilingual-cased"
#BASE_MODEL = "facebook/xlm-v-base"
SCORER_MODEL = "sentence-transformers/LaBSE"
MUSE_PATH = "/home/epsilon/Workbenches/ML_VLM/lrl_adapt/muse"
SCORER_MODEL_TYPE = 'Normal' #Normal, Quality, Size  #1000, 5000, 10000, 20000, 30000 #TODO
SCORER_MODEL_TYPE_PARAM = {'Normal': 'ALL', 'Quality':'LOW', 'Size':5000} #TODO
DEVICE_ID = '1' #TODO
device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
POS_TAGS = ['NOUN', 'ADV'] #NOUN, VERB, ADJ, ADV, INTJ, PROPN
TARGET_LANG = "hi"
LANG_ADAPTER_IDS = {
    "hi": "hi/wiki@ukp",   # Hindi Wikipedia adapter
    "th": "th/wiki@ukp",   # Thai Wikipedia adapter
    "es": "es/wiki@ukp"    # Spanish Wikipedia adapter
}
EPOCHS = 10
TRAIN_BATCH_SIZE = 16
#batch size: 16 is default
RATIO = 0.5 #0.1, 0.3, 0.5, 0.7, Default: 0.5
#CS RATIO: 0.5 is default 
#Below are actual values for reporting
THRESHOLD = 0.9 #TODO 0.40,0.50, 0.55, 0.6,  0.65, 0.70 #0.65-default
#THRESHOLD = 0.0 #TODO - for LLAMA predicted value
TEMP = 7.0 #TODO 0.1,0.3, 0.5, 0.7 # 0.5 - default
#THRESHOLD = 0.90 #TODO 
#TEMP = 0.3  #TODO
PRE_TRAINED_ADAPTER =  False
DICT_NOISE = 0.2
AUGMENT = False #TODO
TARGET_DATA_RATIO = 1.0 #TODO, 0.05, 0.1, 0.3
TRAIN_GROUP = 'ALL' #TODO
LLAMA_ALIGN = "labse" #TODO #"scorer" - tgt language, combination - src+tgt langauges, labse - src langauges
LAYER_GROUPS = {
    'BOTTOM': [4, 5, 6, 7, 8, 9, 10, 11], # Only layers 0, 1, 2, 3 have adapters
    'MIDDLE': [0, 1, 2, 3, 8, 9, 10, 11], # Only layers 4, 5, 6, 7 have adapters
    'TOP':    [0, 1, 2, 3, 4, 5, 6, 7],    # Only layers 8, 9, 10, 11 have adapters
    'ALL':    []                           # All 12 layers have adapters
}
#Hindi (seed: 30) and Thai (seed: 150) for MTOP dataset
#THRESHOLD: 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7
#Threshold: 0.1 (wip), 0.2(wip), 0.5(WIP), 0.7(WIP), 0.9 (WIP) 
#Threshold: 0.43,0,44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50,0.51, 0.52
#Temparature 0.1, 0.3, 1, 2, 5


# --- 2. Deterministic Seeding ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

## LLAMA based scorer model
class LlamaScorePredictorModel(nn.Module):
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float32):
        super().__init__()
        print(f"Initializing Llama Score Predictor via: {model_id}")
        torch_dtype = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype,
            #device_map="auto"
            device_map = device
        )
        self.model.eval()
        self.device = self.model.device

    # ==========================================
    def _get_batch_cosine_scores(self, prompts):
        """Processes an array of prompts concurrently in a single parallel step."""
        formatted_prompts = []
        for prompt_text in prompts:
            messages = [
                {
                    "role": "system", 
                    "content": "You are a precise cross-lingual evaluation metric. Your job is to output exactly one single floating-point cosine similarity number bounded strictly between -1.0 and 1.0. Do not provide commentary."
                },
                {"role": "user", "content": prompt_text}
            ]
            formatted_prompts.append(
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            
        inputs = self.tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(self.device)
        input_lengths = inputs.input_ids.shape[1]
        
        output_tokens = self.model.generate(
            **inputs,
            max_new_tokens=10, 
            do_sample=False,
            temperature=None, 
            top_p=None,       
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        scores = []
        for idx, tokens in enumerate(output_tokens):
            generated_tokens = tokens[input_lengths:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            match = re.search(r"-?\d*\.\d+|-?\d+", response)
            if match:
                try:
                    val = float(match.group())
                    scores.append(min(max(val, -1.0), 1.0))
                except ValueError:
                    scores.append(0.0000)
            else:
                scores.append(0.0000)
        return scores

    # ==========================================
    # CHANGE 3 GOES DIRECTLY AFTER IT
    # ==========================================
        
        # --- Replace your current forward method with this ---
    def forward(self, batch_data, mode="combination"):
        if mode not in ["labse", "scorer", "combination"]:
            raise ValueError("Invalid mode select. Choose from 'labse', 'scorer', or 'combination'.")
            
        if not batch_data:
            return []

        # Step 1: Assemble your prompt lists quickly in memory (no GPU load yet)
        labse_prompts = []
        scorer_prompts = []
        
        for item in batch_data:
            src_sentence = item["src_sentence"]
            cs_sentence = item["cs_sentence"]
            target_lang = item["target_lang"].upper()
            
            if mode in ["labse", "combination"]:
                labse_prompts.append(
                    f"Rate the semantic cosine similarity between the original English sentence and its code-switched version.\n"
                    f"Original English: '{src_sentence}'\n"
                    f"Code-Switched: '{cs_sentence}'\n"
                    f"Predict a cosine similarity score between -1.0 and 1.0. Output only the single number:"
                )
            if mode in ["scorer", "combination"]:
                scorer_prompts.append(
                    f"Predict the cosine similarity alignment of this code-switched sentence with native target language text structures in {target_lang}.\n"
                    f"Sentence: '{cs_sentence}'\n"
                    f"Predict a score between -1.0 and 1.0. Output only the number:"
                )

        # Step 2: Execute parallel evaluation batches
        with torch.no_grad():
            if mode == "labse":
                return self._get_batch_cosine_scores(labse_prompts)
                
            elif mode == "scorer":
                return self._get_batch_cosine_scores(scorer_prompts)
                
            elif mode == "combination":
                # These two steps calculate scores for all items in parallel!
                scores_labse = self._get_batch_cosine_scores(labse_prompts)
                scores_scorer = self._get_batch_cosine_scores(scorer_prompts)
                
                final_scores = []
                for l_score, s_score in zip(scores_labse, scores_scorer):
                    if l_score < 0.30:
                        final_score = -1.0 if l_score < 0.0 else l_score * 0.1
                    else:
                        final_score = (l_score * 0.2) + (s_score * 0.8)
                    final_scores.append(final_score)
                    
                return final_scores 

# --- 3. Scorer Architecture ---
class ScorePredictorModel(nn.Module):
    def __init__(self, model_name, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name, config=config)
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value"],
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.base_model = get_peft_model(self.base_model, lora_config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.score_regressor = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)
        
        # FIXED: Return the Tensor directly so torch.sigmoid() works in batch_code_switch
        score_logits = self.score_regressor(cls_output).squeeze(-1)
        return score_logits
    
class PureLaBSEModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Loads raw LaBSE without any LoRA adapters or custom regression heads
        self.base_model = AutoModel.from_pretrained(model_name)

    def forward(self, src_input_ids, src_attention_mask, cs_input_ids, cs_attention_mask):
        # 1. Get embeddings for the source (English) text
        src_outputs = self.base_model(input_ids=src_input_ids, attention_mask=src_attention_mask)
        src_emb = src_outputs.last_hidden_state[:, 0, :] # [CLS] token
        
        # 2. Get embeddings for the Code-Switched text
        cs_outputs = self.base_model(input_ids=cs_input_ids, attention_mask=cs_attention_mask)
        cs_emb = cs_outputs.last_hidden_state[:, 0, :] # [CLS] token
        
        # 3. L2 Normalize both embeddings
        src_emb = F.normalize(src_emb, p=2, dim=1)
        cs_emb = F.normalize(cs_emb, p=2, dim=1)
        
        # 4. Compute and return Cosine Similarity
        sim_scores = torch.sum(src_emb * cs_emb, dim=1)
        return sim_scores

def load_cs_resources(l1, l2, path):
    cs_dict = defaultdict(list)
    file_path = f"{path}/{l1}-{l2}.txt"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2: continue
                cs_dict[parts[0].lower()].append(parts[1].lower())
    except Exception as e:
        print(f"(!) Dictionary Error: {e}")
    return cs_dict

# --- 4. Batch Code-Switching Logic ---
def batch_code_switch(texts, ratio, cs_dict, scorer, tokenizer, mode):

    if mode == 'english':
        return texts
    
    if mode == "random":
        final_texts = []
        for text in texts:
            tokens = word_tokenize(text)
            swappable = [i for i, w in enumerate(tokens) if w.lower() in cs_dict]
            if not swappable:
                final_texts.append(text); continue
            k = min(max(1, int(ratio * len(tokens))), len(swappable))
            for idx in random.sample(swappable, k):
                tokens[idx] = cs_dict[tokens[idx].lower()][0]
            final_texts.append(' '.join(tokens))
        return final_texts
    
    # BASELINE 2: POS-CONSTRAINED CS (For ARR Analysis Point 1)
    if mode == "pos":
        final_texts = []
        for text in texts:
            tokens = word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens, tagset='universal')
            swappable = [i for i, (w, tag) in enumerate(pos_tags) 
                         if w.lower() in cs_dict and tag in POS_TAGS]
            if not swappable: final_texts.append(text); continue
            k = min(max(1, int(ratio * len(tokens))), len(swappable))
            for idx in random.sample(swappable, k):
                tokens[idx] = cs_dict[tokens[idx].lower()][0]
            final_texts.append(' '.join(tokens))
        return final_texts

    if mode == 'src_alignment' or mode == 'src_tgt_alignment':
        pure_labse_model = PureLaBSEModel(SCORER_MODEL).to(device)

    if mode == 'llama_scorer_tgt':
        llama_model = LlamaScorePredictorModel().to(device)
        print('******llama model loaded succssfully******')

    all_candidates, src_candidates, metadata = [], [], []
    for text in texts:
        tokens = word_tokenize(text)
        swappable = [i for i, w in enumerate(tokens) if w.lower() in cs_dict]
        if not swappable:
            metadata.append(None); continue
        
        k = min(max(1, int(ratio * len(tokens))), len(swappable))
        cands = []
        for idx in swappable:
            trans = cs_dict[tokens[idx].lower()][0]
            temp = list(tokens); temp[idx] = trans
            sent = ' '.join(temp)
            cands.append({"idx": idx, "trans": trans, "sent": sent})
            all_candidates.append(sent)
            src_candidates.append(text)
        metadata.append({"tokens": tokens, "k": k, "cands": cands})

    #print(metadata)
    if not all_candidates: return texts

    all_scores = []
    scorer.eval()
    print(f'Lenght of all_candidates: {len(all_candidates)}, texts is {len(src_candidates)}')
    for i in range(0, len(all_candidates), 128):
        b_slice = all_candidates[i : i + 128]
        a_slice = src_candidates[i : i + 128] 
        inputs = tokenizer(b_slice, return_tensors="pt", padding=True, truncation=True).to(device)
        srcs = tokenizer(a_slice, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            # This line now works because scorer returns a Tensor
            #print(scorer(inputs.input_ids, inputs.attention_mask))
            if mode == 'src_alignment':
                #src_input_ids, src_attention_mask, cs_input_ids, cs_attention_mask
                scores = pure_labse_model(
                src_input_ids=inputs.input_ids,
                src_attention_mask=inputs.attention_mask,
                cs_input_ids=srcs.input_ids,
                cs_attention_mask=srcs.attention_mask
            ).cpu().tolist()
                
            elif mode == 'llama_scorer_tgt':
                sample_batch = [{"src_sentence": src_sent,"cs_sentence": cs_sent, "target_lang": TARGET_LANG}
                for src_sent, cs_sent in zip(a_slice, b_slice)
                ]
                scores = llama_model(sample_batch, mode=LLAMA_ALIGN) #TODO
                print(f'******llama predicted scores******: {scores}')


            elif mode == 'tgt_alignment':
                #src_input_ids, src_attention_mask, cs_input_ids, cs_attention_mask
                scores = torch.sigmoid(scorer(inputs.input_ids, inputs.attention_mask)).cpu().tolist()  

            elif mode == 'src_tgt_alignment':
                #src_input_ids, src_attention_mask, cs_input_ids, cs_attention_mask
                scores_labse = pure_labse_model(
                src_input_ids=inputs.input_ids,
                src_attention_mask=inputs.attention_mask,
                cs_input_ids=srcs.input_ids,
                cs_attention_mask=srcs.attention_mask
            ).cpu().tolist()
                
                scores_scorer = torch.sigmoid(scorer(inputs.input_ids, inputs.attention_mask)).cpu().tolist()
                
                if isinstance(scores_scorer[0], list):
                    scores_scorer = [item[0] for item in scores_scorer]
                else:
                    scores_scorer = scores_scorer
                scores = [(a + b) / 2.0 for a, b in zip(scores_labse, scores_scorer)]
                
                print("Breakdown of [LaBSE, Scorer] -> Final Average:")
                for a, b, final, sent in zip(scores_labse, scores_scorer, scores, b_slice):
                    print(f"Sent: {sent}, LaBSE: {a:.4f} + Scorer: {b:.4f} ==> Avg: {final:.4f}")

            else:
                scores = torch.sigmoid(scorer(inputs.input_ids, inputs.attention_mask)).cpu().tolist()
            all_scores.extend(scores)

    final_texts, ptr = [], 0
    for m in metadata:
        if m is None: final_texts.append(""); continue
        
        valid = []
        for c in m["cands"]:
            print(f"Filtering:: {all_candidates[ptr]}:{all_scores[ptr]}")
            c["score"] = all_scores[ptr]; ptr += 1
            if c["score"] >= THRESHOLD: valid.append(c)
        
        if not valid: 
            final_texts.append(' '.join(m["tokens"])); continue
        #print(f"Length of valid:{len(valid)}, length of swappable: {len(swappable)}")    
        scores_arr = np.array([v["score"] for v in valid])
        probs = np.exp(scores_arr / TEMP) / np.sum(np.exp(scores_arr / TEMP))
        chosen = np.random.choice(len(valid), size=min(m["k"], len(valid)), replace=False, p=probs)
        res_tokens = m["tokens"]
        ##print(chosen)
        for c_idx in chosen:
            res_tokens[valid[c_idx]["idx"]] = valid[c_idx]["trans"]
        final_texts.append(' '.join(res_tokens))
            
    #print("*"*100)

    
    return final_texts


def batch_code_switch_issue(texts, ratio, cs_dict, scorer, tokenizer, mode, 
                      device="cuda", threshold=0.5, temp=0.05):
    """
    Args:
        texts: List of English sentences.
        ratio: Fraction of swappable words to actually swap.
        cs_dict: Dictionary {english_word: [hindi_word]}.
        scorer: The ScorePredictorModel.
        tokenizer: The AutoTokenizer.
        mode: 'english', 'random', or 'model'.
    """
    if mode == 'english':
        return texts
    
    # --- Mode: Random Baseline ---
    if mode == "random":
        final_texts = []
        for text in texts:
            tokens = word_tokenize(text)
            swappable = [i for i, w in enumerate(tokens) if w.lower() in cs_dict]
            if not swappable:
                final_texts.append(text); continue
            
            k = min(max(1, int(ratio * len(tokens))), len(swappable))
            for idx in random.sample(swappable, k):
                tokens[idx] = cs_dict[tokens[idx].lower()][0]
            final_texts.append(' '.join(tokens))
        return final_texts

    # --- Mode: Scorer-Based (Model) ---
    all_candidates = []
    metadata = [] # Stores mapping info to reconstruct sentences later

    for text in texts:
        tokens = word_tokenize(text)
        swappable_indices = [i for i, w in enumerate(tokens) if w.lower() in cs_dict]
        
        if not swappable_indices:
            metadata.append(None)
            continue
        
        # Calculate how many words we want to swap for this specific sentence
        k = min(max(1, int(ratio * len(tokens))), len(swappable_indices))
        
        sentence_candidates = []
        for idx in swappable_indices:
            translation = cs_dict[tokens[idx].lower()][0]
            # Create a candidate sentence where only THIS word is swapped
            temp_tokens = list(tokens)
            temp_tokens[idx] = translation
            candidate_sent = ' '.join(temp_tokens)
            
            sentence_candidates.append({
                "idx": idx, 
                "trans": translation, 
                "sent": candidate_sent
            })
            all_candidates.append(candidate_sent)
            
        metadata.append({
            "tokens": tokens, 
            "k": k, 
            "candidates": sentence_candidates
        })

    if not all_candidates:
        return texts

    # --- Batch Inference ---
    all_scores = []
    scorer.eval()
    batch_size = 64 # Adjust based on GPU memory
    
    with torch.no_grad():
        for i in range(0, len(all_candidates), batch_size):
            batch_texts = all_candidates[i : i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Use the score_logits key from your model's output
            outputs = scorer(**inputs)
            logits = outputs["score_logits"]
            
            # Apply Sigmoid to bound scores between 0 and 1
            scores = torch.sigmoid(logits).cpu().numpy().tolist()
            all_scores.extend(scores)

    # --- Final Selection with Min-Max & Softmax ---
    final_texts = []
    score_ptr = 0
    
    for m in metadata:
        if m is None:
            # If no swappable words were found, return original or empty
            final_texts.append("") # Or handle as original text
            continue
        
        # Attach scores to candidates
        valid_candidates = []
        current_sentence_scores = []
        
        for cand in m["candidates"]:
            score = all_scores[score_ptr]
            score_ptr += 1
            if score >= threshold:
                cand["score"] = score
                valid_candidates.append(cand)
                current_sentence_scores.append(score)
        
        if not valid_candidates:
            final_texts.append(' '.join(m["tokens"]))
            continue

        # --- Min-Max Scaling Logic ---
        # This stretches scores so the best candidate is 1.0 and worst is 0.0
        scores_arr = np.array(current_sentence_scores)
        s_min, s_max = scores_arr.min(), scores_arr.max()
        
        if s_max > s_min:
            normalized_scores = (scores_arr - s_min) / (s_max - s_min)
        else:
            normalized_scores = np.ones_like(scores_arr)

        # --- Softmax Selection ---
        exp_scores = np.exp(normalized_scores / temp)
        probs = exp_scores / np.sum(exp_scores)
        
        # Select 'k' candidates based on the distribution
        num_to_select = min(m["k"], len(valid_candidates))
        chosen_indices = np.random.choice(
            len(valid_candidates), 
            size=num_to_select, 
            replace=False, 
            p=probs
        )
        
        # Apply the chosen swaps to the original tokens
        result_tokens = list(m["tokens"])
        for idx in chosen_indices:
            target_swap = valid_candidates[idx]
            result_tokens[target_swap["idx"]] = target_swap["trans"]
            
        final_texts.append(' '.join(result_tokens))

    return final_texts

# --- 5. MTOP Experiment Runner ---
def run_mtop_experiment(seed, mode):
    set_seed(seed)
    full_ds = load_dataset("WillHeld/mtop")
    
    all_intents = sorted(list(set(full_ds["train_en"]["intent"]) | set(full_ds[f"test_{TARGET_LANG}"]["intent"])))
    num_labels = len(all_intents)
    class_feature = ClassLabel(names=all_intents)
    
    train_ds_en = full_ds["train_en"].cast_column("intent", class_feature)
    if AUGMENT:
        train_ds_tgt = full_ds[f"train_{TARGET_LANG}"].cast_column("intent", class_feature)
        num_tgt_to_include = int(len(train_ds_en) * TARGET_DATA_RATIO)
        train_ds_tgt = train_ds_tgt.shuffle(seed=seed)
        train_ds_tgt_subset = train_ds_tgt.select(range(min(num_tgt_to_include, len(train_ds_tgt))))
    test_ds_tgt = full_ds[f"test_{TARGET_LANG}"].cast_column("intent", class_feature)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    cs_dict = load_cs_resources("en", TARGET_LANG, MUSE_PATH)
    
    scorer = ScorePredictorModel(SCORER_MODEL).to(device)

    if SCORER_MODEL_TYPE == 'Normal':
        al_path = f"/home/epsilon/Workbenches/ML_VLM/lrl_adapt/scripts/best_model_en-{TARGET_LANG}_score_prediction.pth"

    elif SCORER_MODEL_TYPE == 'Quality':
        al_path = f"/home/epsilon/Workbenches/ML_VLM/lrl_adapt/scripts/best_model_en-{TARGET_LANG}_score_prediction_ALL_{SCORER_MODEL_TYPE_PARAM['Quality']}.pth"

    elif SCORER_MODEL_TYPE == 'Size':
        al_path = f"/home/epsilon/Workbenches/ML_VLM/lrl_adapt/scripts/best_model_en-{TARGET_LANG}_score_prediction_{SCORER_MODEL_TYPE_PARAM['Size']}_ALL.pth"

    print(f"****Loaded model path: {al_path}")

    if os.path.exists(al_path):
        print(f"[INFO] Loading trained weights from: {al_path}")
        state_dict = torch.load(al_path, map_location=device)
        
        # Robust loading: handles potential key mismatches from LoRA/PEFT wrapping
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("regressor.", "score_regressor.") # Alignment check
            new_state_dict[new_key] = v
            
        #scorer.load_state_dict(new_state_dict, strict=False)
        scorer.load_state_dict(state_dict)
    else:
        print(f"[WARN] Alignment weights not found. Using raw LaBSE.")
    scorer.eval()

    def process_data(examples, is_train=True):
        texts = examples["utterance"]
        if is_train:
            texts = batch_code_switch(texts, RATIO, cs_dict, scorer, tokenizer, mode)
        enc = tokenizer(texts, padding="max_length", truncation=True, max_length=64)
        enc["label"] = examples["intent"]
        return enc
    
    def process_tgt_pure(examples):
        # DO NOT code-switch. Just tokenize.
        enc = tokenizer(examples["utterance"], padding="max_length", truncation=True, max_length=64)
        enc["label"] = examples["intent"]
        return enc

    train_ds = train_ds_en.map(lambda x: process_data(x, True), batched=True, batch_size=256, remove_columns=train_ds_en.column_names, load_from_cache_file=False)
    test_ds = test_ds_tgt.map(lambda x: process_data(x, False), batched=True, batch_size=256, remove_columns=test_ds_tgt.column_names, load_from_cache_file=False)
    if AUGMENT:
        train_tgt_mapped = train_ds_tgt_subset.map(process_tgt_pure, batched=True, batch_size=256, remove_columns=train_ds_tgt.column_names)
        train_ds = concatenate_datasets([train_ds, train_tgt_mapped]).shuffle(seed=seed)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))
    test_loader = DataLoader(test_ds, batch_size=32, collate_fn=DataCollatorWithPadding(tokenizer))

    model = AutoAdapterModel.from_pretrained(BASE_MODEL)
    config = SeqBnConfig(leave_out= LAYER_GROUPS[TRAIN_GROUP]) #TODO

    if not PRE_TRAINED_ADAPTER:

        #model.add_adapter("intent_adapter", SeqBnConfig())
        model.add_adapter("intent_adapter", config)
        model.add_classification_head("intent_head", num_labels=num_labels)
        model.train_adapter("intent_adapter")
        model.set_active_adapters("intent_adapter")
        model.to(device)

    else:
        model = AutoAdapterModel.from_pretrained(BASE_MODEL)
    
        # 1. Load the pre-trained Language Adapter (automatically kept frozen)
        lang_adapter_name = LANG_ADAPTER_IDS.get(TARGET_LANG.lower())
        if lang_adapter_name:
            print(f"[INFO] Loading language adapter: {lang_adapter_name}")
            loaded_lang_name = model.load_adapter(lang_adapter_name)
        else:
            raise ValueError(f"No pre-trained adapter mapped for TARGET_LANG: {TARGET_LANG}")

        # 2. Add the Task Adapter
        config = SeqBnConfig(leave_out=LAYER_GROUPS[TRAIN_GROUP])
        model.add_adapter("intent_adapter", config)
        
        # 3. Add Classification Head
        model.add_classification_head("intent_head", num_labels=num_labels)
        
        # 4. Set only the Task Adapter and Head to be trainable
        model.train_adapter("intent_adapter")
        
        # 5. Stack them: [Language Adapter -> Task Adapter]
        model.set_active_adapters(Stack(loaded_lang_name, "intent_adapter"))
        model.to(device)    

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fct = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc=f"Seed {seed} | Mode {mode} | Ep {epoch}", leave=False):
            lbl = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = loss_fct(model(**batch).logits, lbl)
            loss.backward(); optimizer.step(); optimizer.zero_grad()

        model.eval()
        preds, golds = [], []
        for batch in test_loader:
            lbl = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**batch).logits
                preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                golds.extend(lbl.numpy())
        
        acc = accuracy_score(golds, preds)
        if acc > best_acc: best_acc = acc

    del model, optimizer, scorer, train_loader, test_loader
    torch.cuda.empty_cache(); gc.collect()
    
    return best_acc * 100

# --- 6. Execution Loop ---
if __name__ == "__main__":
    #SEEDS = [i * 10 for i in range(1, 21)] 
    SEEDS = [10, 20, 30]#TODO
    results_list = []

    for s in SEEDS:
        res = {"seed": s}
        #["random", "selective_best"]
        #configs = ["random", "selective_best", 'english', 'pos', 'src_alignment', 'src_tgt_alignment']
        configs = ['selective_best']
        #configs = ['llama_scorer_tgt']
        for m in configs:
            final_acc = run_mtop_experiment(s, m)
            res[m] = final_acc
            print(f"| Seed {s} | {m}: {final_acc:.2f}% |")
        results_list.append(res)

        if len(configs) == 1 and configs[0]=='english':
            pd.DataFrame(results_list).to_csv(f"mtop_20seeds_english_{TARGET_LANG}.csv", index=False)

        elif len(configs) == 1 and configs[0]=='pos': #TODO
            pd.DataFrame(results_list).to_csv(f"mtop_20seeds_pos_AUGMENT_{RATIO}_{TARGET_DATA_RATIO}_{TARGET_LANG}_{POS_TAGS}.csv", index=False)

        elif len(configs) == 1 and configs[0]=='random':
            pd.DataFrame(results_list).to_csv(f"mtop_20seeds_random_{RATIO}_{TARGET_DATA_RATIO}_{TARGET_LANG}.csv", index=False)
    
        elif len(configs) == 1 and configs[0]=='src_alignment':
            pd.DataFrame(results_list).to_csv(f"mtop_20seeds_random_SRC_ALIGNMENT_{TARGET_LANG}_{EPOCHS}.csv", index=False)

        elif len(configs) == 1 and configs[0]=='tgt_alignment':
            pd.DataFrame(results_list).to_csv(f"mtop_20seeds_random_TGT_ALIGNMENT_{TARGET_LANG}_{EPOCHS}.csv", index=False)


        elif len(configs) == 1 and configs[0]=='src_tgt_alignment':
            pd.DataFrame(results_list).to_csv(f"mtop_20seeds_random_SRC_TGT_ALIGNMENT_{TARGET_LANG}_{EPOCHS}.csv", index=False)


        elif len(configs) == 1 and configs[0]=='selective_best1':
            pd.DataFrame(results_list).to_csv(f"mtop_20seeds_random_selective_best_{TARGET_LANG}_{EPOCHS}_{SCORER_MODEL_TYPE}_{SCORER_MODEL_TYPE_PARAM[SCORER_MODEL_TYPE]}.csv", index=False)

        elif len(configs) == 1 and configs[0]=='llama_scorer_tgt':
            pd.DataFrame(results_list).to_csv(f"mtop_20seeds_selective_best_{configs[0]}_{LLAMA_ALIGN}_{TARGET_LANG}.csv", index=False)

        else:    
            pd.DataFrame(results_list).to_csv(f"mtop_20seeds_{TARGET_LANG}_{THRESHOLD}_{TEMP}.csv", index=False)
            #pd.DataFrame(results_list).to_csv(f"mtop_20seeds_epochs_{TARGET_LANG}_{EPOCHS}.csv", index=False)
    '''
    df = pd.DataFrame(results_list)
    t_stat, p_val = stats.ttest_ind(df["selective_best"], df["random"])
    
    print("\n" + "="*50)
    print(f"FINAL STATISTICAL RESULTS FOR {TARGET_LANG}")
    print("="*50)
    print(df[["random", "selective_best"]].describe().loc[['mean', 'std', 'min', 'max']])
    print("-" * 50)
    print(f"T-statistic: {t_stat:.4f} | P-value: {p_val:.4f}")
    '''