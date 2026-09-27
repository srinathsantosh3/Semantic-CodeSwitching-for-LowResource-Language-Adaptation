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
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from torch.utils.data import DataLoader
from adapters import AutoAdapterModel, SeqBnConfig
from nltk import word_tokenize
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
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
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
from nltk import pos_tag

# --- 1. Global Configuration ---
BASE_MODEL = "xlm-roberta-base"
SCORER_MODEL = "sentence-transformers/LaBSE"
MUSE_PATH = "/home/epsilon/Workbenches/ML_VLM/lrl_adapt/muse"
DEVICE_ID = '2'#TODO
device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")

TRAIN_SUBSET = 50000#TODO 
TARGET_LANG = "es" #TODO
EPOCHS = 10
RATIO = 0.5 
TEMP = 0.3 
#THRESHOLD = 0.65 
THRESHOLD = 0.0 #TODO - for LLAMA predicted value
POS_TAGS = ['NOUN', 'PROPN']
LLAMA_ALIGN = "scorer" #TODO #"scorer" - tgt language, combination - src+tgt langauges, labse - src langauges
SCORER_MODEL_TYPE = 'Normal' #Normal, Quality - LOW, MEDIUM, HIGH, Size - 1000 ,5000, 10000, 20000, 30000  #TODO
SCORER_MODEL_TYPE_PARAM = {'Normal': 'ALL', 'Quality':'MEDIUM', 'Size':1000} #TODO
#THRESHOLD: 0.45, 0.55, 0.65, 0.75
#Temparature 0.1, 0.3, 1


# --- 2. Deterministic Seeding ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import re
from transformers import AutoModelForCausalLM

class PureLaBSEModel(nn.Module):
    def __init__(self, model_name="sentence-transformers/LaBSE"):
        super().__init__()
        print(f"Initializing Pure LaBSE Scorer via: {model_name}")
        # Load the dedicated LaBSE tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        # Ensure it knows what device it is on
        self.device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
        self.base_model.to(self.device)

    def forward(self, src_texts, cs_texts):
        # 1. Tokenize using the correct LaBSE vocabulary
        src_inputs = self.tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        cs_inputs = self.tokenizer(cs_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)

        # 2. Get embeddings for the source (English) text
        src_outputs = self.base_model(**src_inputs)
        src_emb = src_outputs.last_hidden_state[:, 0, :] # [CLS] token
        
        # 3. Get embeddings for the Code-Switched text
        cs_outputs = self.base_model(**cs_inputs)
        cs_emb = cs_outputs.last_hidden_state[:, 0, :] # [CLS] token
        
        # 4. L2 Normalize both embeddings
        src_emb = F.normalize(src_emb, p=2, dim=1)
        cs_emb = F.normalize(cs_emb, p=2, dim=1)
        
        # 5. Compute and return Cosine Similarity
        sim_scores = torch.sum(src_emb * cs_emb, dim=1)
        return sim_scores

class LlamaScorePredictorModel(nn.Module):
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16):
        super().__init__()
        print(f"Initializing Llama Score Predictor via: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype,
            device_map=device
        )
        self.model.eval()
        self.device = self.model.device

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
        
    def forward(self, batch_data, mode="combination"):
        if mode not in ["labse", "scorer", "combination"]:
            raise ValueError("Invalid mode select. Choose from 'labse', 'scorer', or 'combination'.")
            
        if not batch_data:
            return []

        labse_prompts = []
        scorer_prompts = []
        
        # Tailored explicitly for XNLI Premise/Hypothesis context mappings
        for item in batch_data:
            src_sentence = item["src_sentence"]
            cs_sentence = item["cs_sentence"]
            target_lang = item["target_lang"].upper()
            context_type = item.get("context_type", "sentence") # Can indicate 'premise' or 'hypothesis'
            
            if mode in ["labse", "combination"]:
                labse_prompts.append(
                    f"Rate the semantic cosine similarity between the original English NLI text component and its mutated code-switched version.\n"
                    f"Original English {context_type}: '{src_sentence}'\n"
                    f"Code-Switched {context_type}: '{cs_sentence}'\n"
                    f"Predict a cosine similarity score between -1.0 and 1.0 based on structural context preservation. Output only the single number:"
                )
            if mode in ["scorer", "combination"]:
                scorer_prompts.append(
                    f"Predict the cosine similarity alignment of this code-switched cross-lingual sentence structure with standard native text expressions in {target_lang}.\n"
                    f"NLI Context Text: '{cs_sentence}'\n"
                    f"Predict a structural naturalness compatibility score between -1.0 and 1.0. Output only the number:"
                )

        with torch.no_grad():
            if mode == "labse":
                return self._get_batch_cosine_scores(labse_prompts)
            elif mode == "scorer":
                return self._get_batch_cosine_scores(scorer_prompts)
            elif mode == "combination":
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

# --- 3. Model Definitions ---
class ScorePredictionModel0(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.base.config.hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        return self.regressor(out.last_hidden_state[:, 0]).squeeze(-1)

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
        self.loss_fct_score = nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels_score=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)
        score_logits = self.score_regressor(cls_output).squeeze(-1)
        total_loss = None
        if labels_score is not None:
            total_loss = self.loss_fct_score(score_logits, labels_score.float())
        return {"loss": total_loss, "score_logits": score_logits}

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

# --- 4. Selective Code-Switching Logic (FIXED) ---
def batch_code_switch(texts, ratio, cs_dict, scorer, scorer_labse, tokenizer, temperature=0.3, min_threshold=0.55, mode = 'llama'):
    all_candidates, src_candidates, metadata = [], [], []

    for text in texts:
        tokens = word_tokenize(text)
        swappable = [i for i, w in enumerate(tokens) if w.lower() in cs_dict]
        if not swappable:
            metadata.append(None); continue
        
        k = min(max(1, int(ratio * len(tokens))), len(swappable))
        cands_for_text = []
        for idx in swappable:
            trans = cs_dict[tokens[idx].lower()][0]
            temp_tokens = list(tokens)
            temp_tokens[idx] = trans
            sent_str = ' '.join(temp_tokens)
            cands_for_text.append({"token_idx": idx, "trans": trans, "sent": sent_str})
            all_candidates.append(sent_str)
            src_candidates.append(text)
        metadata.append({"tokens": tokens, "k": k, "cands": cands_for_text})

    if not all_candidates: return texts

    all_scores = []
    scorer.eval()
    batch_chunk_size = 16 if 'llama' in mode else 218
    
    for i in range(0, len(all_candidates), batch_chunk_size):
        b_slice = all_candidates[i : i + batch_chunk_size]
        a_slice = src_candidates[i : i + batch_chunk_size]
        payload_batch = []
        for cand_sent in b_slice:
            payload_batch.append({
                "src_sentence": text, # Captures reference string context tracking
                "cs_sentence": cand_sent,
                "target_lang": TARGET_LANG,
                "context_type": "text component"
            })

        inputs = tokenizer(b_slice, return_tensors="pt", padding=True, truncation=True).to(device)
        srcs = tokenizer(a_slice, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            # FIX: Access ["score_logits"] and use flatten()
            if 'llama' in mode:
                scores = scorer(payload_batch, mode=LLAMA_ALIGN)

            if mode == 'src_alignment':
                
                src_texts = [p["src_sentence"] for p in payload_batch]
                cs_texts = [p["cs_sentence"] for p in payload_batch]
                
                sim_scores = scorer_labse(src_texts, cs_texts)
                scores = sim_scores.cpu().flatten().tolist()

            elif mode == 'src_tgt_alignment':
                src_texts = [p["src_sentence"] for p in payload_batch]
                cs_texts = [p["cs_sentence"] for p in payload_batch]
                
                sim_scores = scorer_labse(src_texts, cs_texts)
                scores_labse = sim_scores.cpu().flatten().tolist()

                outputs = scorer(inputs.input_ids, inputs.attention_mask)
                logits = outputs["score_logits"]
                scores_scorer = torch.sigmoid(logits).cpu().flatten().tolist()
                if isinstance(scores_scorer[0], list):
                    scores_scorer = [item[0] for item in scores_scorer]
                else:
                    scores_scorer = scores_scorer
                scores = [(a + b) / 2.0 for a, b in zip(scores_labse, scores_scorer)]       

            else:
                outputs = scorer(inputs.input_ids, inputs.attention_mask)
                logits = outputs["score_logits"]
                scores = torch.sigmoid(logits).cpu().flatten().tolist()

            all_scores.extend(scores)

    final_texts, score_ptr = [], 0
    for i, m in enumerate(metadata):
        if m is None:
            final_texts.append(texts[i]); continue
            
        current_cands = m["cands"]
        for cand in current_cands:
            #print(f"Filtering:: {all_candidates[score_ptr]}:{all_scores[score_ptr]}")
            cand["score"] = all_scores[score_ptr]; score_ptr += 1
            
        valid_cands = [c for c in current_cands if c["score"] >= min_threshold]
        if not valid_cands:
            final_texts.append(' '.join(m["tokens"])); continue

        scores_arr = np.array([c["score"] for c in valid_cands])
        exp_scores = np.exp(scores_arr / temperature)
        probs = exp_scores / exp_scores.sum()
        
        k = min(m["k"], len(valid_cands))
        chosen_indices = np.random.choice(len(valid_cands), size=k, replace=False, p=probs)
        
        res_tokens = list(m["tokens"])
        for idx in chosen_indices:
            c = valid_cands[idx]
            res_tokens[c["token_idx"]] = c["trans"]
            
        final_texts.append(' '.join(res_tokens))
    return final_texts

from datasets import load_dataset, load_from_disk
def get_xnli_data(lang, split, save_path):
    """Downloads dataset once and reuses local disk copy to avoid 502 errors."""
    if os.path.exists(save_path):
        return load_from_disk(save_path)
    else:
        print(f"(!) Downloading {lang} {split} from Hub (this requires internet)...")
        ds = load_dataset("xnli", lang, split=split)
        ds.save_to_disk(save_path)
        return ds

CACHE_DIR = "./hf_cache"
OS_DATA_PATH_EN = f"{CACHE_DIR}/xnli_en_train"
OS_DATA_PATH_TGT = f"{CACHE_DIR}/xnli_{TARGET_LANG}_test"

# --- 5. Experiment Logic ---
def run_experiment(seed, mode):
    set_seed(seed)
    # Load from disk to bypass HF Hub 502 Errors
    train_ds_en = get_xnli_data("en", "train", OS_DATA_PATH_EN)
    test_ds_tgt = get_xnli_data(TARGET_LANG, "test", OS_DATA_PATH_TGT)
    train_ds_en = load_dataset("xnli", "en", split="train")
    if TRAIN_SUBSET:
        train_ds_en = train_ds_en.shuffle(seed=seed).select(range(TRAIN_SUBSET))
    test_ds_tgt = load_dataset("xnli", TARGET_LANG, split="test")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    cs_dict = load_cs_resources("en", TARGET_LANG, MUSE_PATH)
    if 'llama' not in mode :
        scorer = ScorePredictorModel(SCORER_MODEL).to(device)
        #al_path = f"/home/epsilon/Workbenches/ML_VLM/lrl_adapt/scripts/best_model_en-{TARGET_LANG}_alignment.pth"
        if SCORER_MODEL_TYPE == 'Normal':
            al_path = f"/home/epsilon/Workbenches/ML_VLM/lrl_adapt/scripts/best_model_en-{TARGET_LANG}_score_prediction.pth"

        elif SCORER_MODEL_TYPE == 'Quality':
            al_path = f"/home/epsilon/Workbenches/ML_VLM/lrl_adapt/scripts/best_model_en-{TARGET_LANG}_score_prediction_ALL_{SCORER_MODEL_TYPE_PARAM['Quality']}.pth"

        elif SCORER_MODEL_TYPE == 'Size':
            al_path = f"/home/epsilon/Workbenches/ML_VLM/lrl_adapt/scripts/best_model_en-{TARGET_LANG}_score_prediction_{SCORER_MODEL_TYPE_PARAM['Size']}_ALL.pth"

        
        if os.path.exists(al_path):
            checkpoint = torch.load(al_path, map_location=device)
            # Robust remapping for keys
            fixed_dict = {k.replace("regressor.", "score_regressor."): v for k, v in checkpoint.items()}
            #scorer.load_state_dict(fixed_dict, strict=False)
            scorer.load_state_dict(checkpoint, strict=False)
        scorer.eval()

    else:
        print('****LOADED LLAMA MODEL*****')
        scorer = LlamaScorePredictorModel(model_id="meta-llama/Meta-Llama-3-8B-Instruct")
        scorer.eval()

    if mode in ['src_alignment', 'src_tgt_alignment']:
        scorer_labse = PureLaBSEModel(SCORER_MODEL).to(device)
        scorer_labse.eval()    

    def process_data(batch, is_train=True):
        p, h = batch["premise"], batch["hypothesis"]
        if is_train:

            if mode == "pos":
                def pos_cs(txt):
                    tk = word_tokenize(txt)
                    # Use universal tagset for consistent NOUN/ADJ mapping
                    tags = pos_tag(tk, tagset='universal')
                    # Filter: Word must be in dictionary AND be a Noun or Adjective
                    sw = [i for i, (w, t) in enumerate(tags) 
                          if w.lower() in cs_dict and t in POS_TAGS]
                    if not sw: return txt
                    
                    k = min(len(sw), max(1, int(RATIO * len(tk))))
                    for i in random.sample(sw, k):
                        tk[i] = cs_dict[tk[i].lower()][0]
                    return ' '.join(tk)
                p, h = [pos_cs(x) for x in p], [pos_cs(x) for x in h]

            elif mode == "selective_best" or 'llama' in mode :
                p = batch_code_switch(p, RATIO, cs_dict, scorer,scorer_labse, tokenizer, TEMP, THRESHOLD, mode)
                h = batch_code_switch(h, RATIO, cs_dict, scorer,scorer_labse, tokenizer, TEMP, THRESHOLD, mode)
            elif mode == 'english':
                pass    
            elif mode == "random":
                def rand_cs(txt):
                    tk = word_tokenize(txt)
                    sw = [i for i, w in enumerate(tk) if w.lower() in cs_dict]
                    if not sw: return txt
                    for i in random.sample(sw, min(len(sw), max(1, int(RATIO*len(tk))))):
                        tk[i] = cs_dict[tk[i].lower()][0]
                    return ' '.join(tk)
                p, h = [rand_cs(x) for x in p], [rand_cs(x) for x in h]

        enc = tokenizer(p, h, padding="max_length", truncation=True, max_length=128)
        enc["label"] = batch["label"]
        return enc

    train_ds = train_ds_en.map(lambda x: process_data(x, True), batched=True, batch_size=256, remove_columns=train_ds_en.column_names, load_from_cache_file=False)
    test_ds = test_ds_tgt.map(lambda x: process_data(x, False), batched=True, batch_size=256, remove_columns=test_ds_tgt.column_names, load_from_cache_file=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))
    test_loader = DataLoader(test_ds, batch_size=64, collate_fn=DataCollatorWithPadding(tokenizer))

    model = AutoAdapterModel.from_pretrained(BASE_MODEL)
    model.add_adapter("xnli_adapter", SeqBnConfig())
    model.add_classification_head("xnli_head", num_labels=3)
    model.train_adapter("xnli_adapter")
    model.set_active_adapters("xnli_adapter")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fct = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for b in tqdm(train_loader, desc=f"Seed {seed} | {mode} | Ep {epoch}", leave=False):
            lbl = b.pop("labels").to(device)
            b = {k: v.to(device) for k, v in b.items()}
            loss = loss_fct(model(**b).logits, lbl)
            loss.backward(); optimizer.step(); optimizer.zero_grad()

        model.eval()
        preds, golds = [], []
        for b in test_loader:
            lbl = b.pop("labels")
            b = {k: v.to(device) for k, v in b.items()}
            with torch.no_grad():
                logits = model(**b).logits
                preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                golds.extend(lbl.numpy())
        
        acc = accuracy_score(golds, preds)
        if acc > best_acc: best_acc = acc
        
    del model, scorer; torch.cuda.empty_cache(); gc.collect()
    return best_acc * 100

# --- 6. Execution Loop ---
if __name__ == "__main__":
    SEEDS = [i * 10 for i in range(1, 21)]
    #SEEDS = [i * 10 for i in range(8, 21)]
    #SEEDS = [10]
    results = []
    for s in SEEDS:
        seed_res = {"seed": s}
        #configs  = ["selective_best", "random", 'pos']
        configs = ['pos'] 
        #configs = ['llama_src'] #TODO
        for m in configs:
            acc = run_experiment(s, m)
            seed_res[m] = acc
            print(f"| Seed: {s} | Mode: {m} | Acc: {acc:.2f}% |")
        results.append(seed_res)
        #pd.DataFrame(results).to_csv(f"xnli_seeds_{TARGET_LANG}_{RATIO}_{TEMP}_{THRESHOLD}.csv", index=False)
        if len(configs)==1 and  configs[0]=='english':
            pd.DataFrame(results).to_csv(f"xnli_seeds_english_{TARGET_LANG}.csv", index=False)
        elif len(configs)==1 and  configs[0]=='pos':
            pd.DataFrame(results).to_csv(f"xnli_seeds_pos_{TARGET_LANG}_{POS_TAGS}.csv", index=False)  
        elif len(configs)==1 and  configs[0]=='selective_best':
            pd.DataFrame(results).to_csv(f"xnli_seeds_pos_{TARGET_LANG}_{SCORER_MODEL_TYPE}_{SCORER_MODEL_TYPE_PARAM[SCORER_MODEL_TYPE]}.csv", index=False)    
        
        elif len(configs) == 1 and configs[0]=='src_alignment':
            pd.DataFrame(results).to_csv(f"xnli_seeds_SRC_ALIGNMENT_{TARGET_LANG}_{EPOCHS}.csv", index=False)

        elif len(configs) == 1 and configs[0]=='tgt_alignment':
            pd.DataFrame(results).to_csv(f"xnli_seeds_TGT_ALIGNMENT_{TARGET_LANG}_{EPOCHS}.csv", index=False)


        elif len(configs) == 1 and configs[0]=='src_tgt_alignment':
            pd.DataFrame(results).to_csv(f"xnli_seeds_SRC_TGT_ALIGNMENT_{TARGET_LANG}_{EPOCHS}.csv", index=False)

        
        elif len(configs)==1 and  'llama' in configs[0]:
            pd.DataFrame(results).to_csv(f"xnli_seeds_llama_{LLAMA_ALIGN}_{TARGET_LANG}.csv", index=False)  
              
        else:    
            #pd.DataFrame(results).to_csv(f"xnli_seeds_{TARGET_LANG}_{THRESHOLD}_{TEMP}.csv", index=False)
            pd.DataFrame(results).to_csv(f"xnli_seeds_{TARGET_LANG}.csv", index=False)