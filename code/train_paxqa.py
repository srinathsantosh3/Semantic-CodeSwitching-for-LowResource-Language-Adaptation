import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoConfig
from adapters import AutoAdapterModel, SeqBnConfig
import random
import numpy as np
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
import pandas as pd
from datasets import load_from_disk
from nltk import word_tokenize
import nltk
# Set Seeds for Reproducibility


BASE_MODEL = "xlm-roberta-base"
SCORER_MODEL = "sentence-transformers/LaBSE"
MUSE_PATH = "/home/epsilon/Workbenches/ML_VLM/lrl_adapt/muse"
SRC_LANG = 'ru'#en: default
TARGET_LANG = "ru" #ru,zh,ar
EPOCHS = 10
TRAIN_BATCH_SIZE = 16
TRAIN_SIZE = 10000
BATCH_SIZE = 16
#batch size: 16 is default
RATIO = 0.5 #0.1, 0.3, 0.5, 0.7
#CS RATIO: 0.5 is default 
#Below are actual values for reporting
#threshold: 0.5, temperature: 0.3 are the best parameters based on experiments
THRESHOLD = 0.5 #TODO 0.45, 0.5, 0.55, 0.6, 0.65, 
TEMP = 0.3 #TODO 0.1, 0.3, 0.7, 1
POS_TAGS = ['NOUN', 'VERB']
##ADJ, ADV,INTJ, NOUN,PROPN, VERB
DEVICE_ID = '1'
DEVICE = f"cuda:{DEVICE_ID}"
#THRESHOLD = 0.65 #TODO 
#TEMP = 0.3 #TODO

SOURCE_DICT = {}
SOURCE_DICT['ar'] = 'gale'
SOURCE_DICT['ru'] = 'nc'
SOURCE_DICT['zh'] = 'nc'

#en_ar_zh[verb](galve)
#en_ru_zh[propn][WIP] nc
#en_zh_ru[verb][completed] nc
# en_ru_ar gv[completed]
# --- 1. Semantic Scorer Initialization ---
# Used for the 'semantic' baseline comparison


# --- 3. Scorer Architecture ---

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

#semantic_scorer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(DEVICE)
semantic_scorer = ScorePredictorModel(SCORER_MODEL).to(DEVICE)


def evaluate_model(model,test_loader, lang,tokenizer,device):

    model.set_active_adapters([lang, "qa-task"])        
    model.eval()
    print(f" Evaluation adapter summary: {model.adapter_summary()}")
    accuracy = []
    total = 0
    for batch in test_loader:
        #from evaluate import load
        #squad_metric = load("squad_v2")

        print(f"evaluation_language is {lang}")
        batch = prepare_train_features(batch,tokenizer)
        '''
        questions = [q for q in batch["question"]]
        contexts = [c for c in batch["context"]]

        batch = tokenizer.batch_encode_plus( questions, contexts, padding='max_length',truncation="only_second", return_tensors='pt' ) 
        #encoding = tokenizer.batch_encode_plus( qa_pairs, padding='max_length',truncation="only_second", return_tensors='pt',return_offsets_mapping=True ) 
        '''
        batch['input_ids'] = torch.tensor(batch["input_ids"] ).to(device)
        batch['attention_mask'] = torch.tensor(batch["attention_mask"] ).to(device)
        
        batch["start_positions"] = torch.tensor(batch["start_positions"]).to(device)
        batch["end_positions"] = torch.tensor(batch["end_positions"]).to(device)

        
        with torch.no_grad():
            outputs = model(**batch)
        #print(f"Outputs post evaluation: {outputs}")
        logits =  outputs.start_logits
        logits_end = outputs.end_logits
        predictions = torch.argmax(logits, dim=1)
        predictions_start = predictions.to('cpu')
        #print(f"start predictions: {predictions}")
        actuals_start = batch['start_positions'].to('cpu')
        actuals_end = batch['end_positions'].to('cpu')
        predictions = torch.argmax(logits_end, dim=1)
        predictions_end = predictions.to('cpu')
        current_list_acc = (actuals_start==predictions_start).tolist()
        #print(f"end predictions: {predictions}")
        #print(f"end actual:{batch['end_positions'].to('cpu')}")
        #print(f"len of predictions:{len(predictions)}, len of labels: {len(labels)}")
        print(f"eval: logits_start: {outputs.start_logits} ")
        print(f"eval: logits_end: {outputs.end_logits} ")
        print(f"eval: actuals{actuals_start}")
        print(f"eval: actuals{predictions_start}")
        print(f"eval: actuals{actuals_end}")
        print(f"eval: actuals{predictions_end}")
        print(f"eval: accuracy{sum(current_list_acc)}")
        print("*"*80)
        #print(f"current correct predictions: {(predictions == labels).sum().item()}")
        total+=len(current_list_acc)
        accuracy+=current_list_acc
        #print(f"current_accuracy:{((predictions == labels).sum().item())/256.0}")

    accuracy = float(sum(accuracy)/ total)
    print(f"accuracy:{accuracy}")
    mean_accuracy,sd = np.average(accuracy), np.std(accuracy)
    return mean_accuracy,sd



def get_semantic_similarity(orig, switched):
    embeddings = semantic_scorer.encode([orig, switched], convert_to_tensor=True)
    return util.cos_sim(embeddings[0], embeddings[1]).item()

# --- 2. Enhanced Code-Switching Engine ---
def apply_codeswitch_prev(text, dictionary,  mode, answer_text=None):
    """
    Handles Random, POS, and Semantic Switching.
    answer_text: If provided, words in the answer are excluded from switching.
    """
    if not dictionary or not text:
        return text

    words = text.split()
    protected_words = set(answer_text.lower().split()) if answer_text else set()
    
    # Identify indices that exist in dictionary and are NOT protected
    valid_indices = [
        i for i, w in enumerate(words) 
        if w.lower() in dictionary and w.lower() not in protected_words
    ]
    
    if not valid_indices:
        return text

    target_count = max(1, int(RATIO * len(words)))
    target_indices = random.sample(valid_indices, min(len(valid_indices), target_count))

    if mode == 'random':
        for idx in target_indices:
            words[idx] = dictionary[words[idx].lower()][0]
        return " ".join(words)

    elif mode == 'pos':
        # This integrates your POS-based filtering logic
        # (Assuming your multi_pos functions are imported as in your snippet)
        # Logic: Only switch if POS matches target (e.g., 'NOUN')
        return " ".join(words) # Placeholder for your specific POS logic

    elif mode == 'semantic':
        # SCS Logic: Generate candidates and pick the one with highest fidelity
        best_sent = " ".join(words)
        best_score = -1.0
        
        for _ in range(3): # Generate 3 variants
            temp_words = text.split()
            current_targets = random.sample(valid_indices, min(len(valid_indices), target_count))
            for idx in current_targets:
                temp_words[idx] = dictionary[words[idx].lower()][0]
            
            candidate = " ".join(temp_words)
            score = get_semantic_similarity(text, candidate)
            
            if score > best_score:
                best_score = score
                best_sent = candidate
        
        return best_sent if best_score > THRESHOLD else text # Threshold filter

    return text

 # --- 4. Batch Code-Switching Logic ---
def batch_code_switch(texts, ratio, cs_dict, scorer, tokenizer, mode):

    device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")

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


    all_candidates, metadata = [], []
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
        metadata.append({"tokens": tokens, "k": k, "cands": cands})

    #print(metadata)
    if not all_candidates: return texts

    all_scores = []
    scorer.eval()
    for i in range(0, len(all_candidates), 128):
        b_slice = all_candidates[i : i + 128]
        inputs = tokenizer(b_slice, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            # This line now works because scorer returns a Tensor
            #print(scorer(inputs.input_ids, inputs.attention_mask))
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
   

class PAXDataset(Dataset):
    def __init__(self, lang,split,data_source, tokenizer):
        #Arabic, Chinese, English
        self.tokenizer = tokenizer
        print(f"split:{split}")
        split = "train" if split == "memory" else split 
        base_path = f"/mnt/MIG_store/Datasets/epsilon/datasets/paxqa/{split}"
        if lang!="en":
            '''
            if data_source=="gale":
                self.dataset = load_from_disk(f"{base_path}/{split}__{data_source}_{lang}_en")
            elif data_source=="nc":
                self.dataset = load_from_disk(f"{base_path}/{split}__{data_source}_zh_en")
            elif data_source=="gv":
                self.dataset = load_from_disk(f"{base_path}/{split}__{data_source}_ar_en")
            '''
                
            self.dataset = load_from_disk(f"{base_path}/{split}__{data_source}_{lang}_en")    
            self.dataset = self.dataset.to_pandas()
        
            is_zero  = self.dataset['answers_src'].apply(lambda x: x['answer_start'][0])==0
            self.dataset = self.dataset[~is_zero]
            is_na_ans_start = self.dataset['answers_src'].apply(lambda x: x['answer_start'][0]).isna()
            self.dataset = self.dataset[~is_na_ans_start]
            len_nonzero  = self.dataset["answers_src"].apply(lambda x: len(tokenizer.tokenize(x['text'][0])))!=0
            self.dataset = self.dataset[len_nonzero]
            self.question = self.dataset["question_src"].tolist()
            self.context = self.dataset["context_src"].tolist()
            self.answer = self.dataset["answers_src"].tolist()
        else:
            if data_source!="gv":
                self.dataset = load_from_disk(f"{base_path}/{split}__{data_source}_zh_en")
            else:
                self.dataset = load_from_disk(f"{base_path}/{split}__{data_source}_ar_en")
            self.dataset = self.dataset.to_pandas()
        
            is_zero  = self.dataset['answers_en'].apply(lambda x: x['answer_start'][0])==0
            self.dataset = self.dataset[~is_zero]
            is_na_ans_start = self.dataset['answers_en'].apply(lambda x: x['answer_start'][0]).isna()
            self.dataset = self.dataset[~is_na_ans_start]
            len_nonzero  = self.dataset["answers_en"].apply(lambda x: len(tokenizer.tokenize(x['text'][0])))!=0
            self.dataset = self.dataset[len_nonzero]
            
            self.question = self.dataset["question_en"].tolist()
            self.context = self.dataset["context_en"].tolist()
            self.answer = self.dataset["answers_en"].tolist()

        self.length = 0
        self.truncate_dataset()
    def   truncate_dataset(self):
        lens_flag = [len(self.tokenizer.tokenize(context))+len(self.tokenizer.tokenize(self.question[i])) <=512 for i,context in enumerate(self.context)]
        self.question =[question for i,question in enumerate(self.question) if lens_flag[i]] 
        self.context = [context for i,context in enumerate(self.context) if lens_flag[i]] 
        self.answer = [answer for i,answer in enumerate(self.answer) if lens_flag[i]] 
        self.length = len(self.answer)

    def __len__(self):
        return min(self.length,TRAIN_SIZE)
    
    def get_max_context_length(self):
        return max([len(x) for x in self.context])
    
    def __getitem__(self, idx):
        try:
            clean_answer = {
            "answer_start": self.answer[idx]["answer_start"][0],
            "text": self.answer[idx]["text"][0]
        }
        except:
            clean_answer = {
            "answer_start": -1,
            "text": ""
        }    
        
        try:
            return {
                
                "question": self.question[idx],
                "answer": clean_answer,
                "context":self.context[idx] 
            }
        except:
            idx = random.randint(0,min(TRAIN_SIZE,self.length)-1)

            return {
                
                "question": self.question[idx],
                "answer": clean_answer,
                "context":self.context[idx] 
            }


def prepare_train_features(examples, tokenizer, max_length=512, doc_stride=128):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_idx = sample_mapping[i]

        # ✅ Corrected answer access
        answer_start_char = examples["answer"]["answer_start"][sample_idx]
        answer_text = examples["answer"]["text"][sample_idx]
        answer_end_char = answer_start_char + len(answer_text)

        # Find context token span
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Check if answer is inside the span
        if not (offsets[token_start_index][0] <= answer_start_char and offsets[token_end_index][1] >= answer_end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_pos = end_pos = None
            for idx in range(token_start_index, token_end_index + 1):
                start, end = offsets[idx]
                if start <= answer_start_char < end:
                    start_pos = idx
                if start < answer_end_char <= end:
                    end_pos = idx
            if start_pos is None or end_pos is None:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                start_positions.append(start_pos)
                end_positions.append(end_pos)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples


# --- 4. Training Engine ---
def train_model(seed,mode):

    set_seed(seed) 
    # Setup Device
    device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
    
    # Initialize Model & Tokenizer
    config = AutoConfig.from_pretrained(BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoAdapterModel.from_pretrained(BASE_MODEL, config=config).to(device)

   
    # Using your get_all_codeswitch function
    
    current_dict = load_cs_resources('en', TARGET_LANG, MUSE_PATH)
    # 1. Add Task Adapter (QA)
    model.add_adapter("qa-task", config="pfeiffer")
    model.add_qa_head("qa-task", num_labels=2)

    
    print(f"Training on: {TARGET_LANG} with {mode} CS")
    
    # Add/Load Language Adapter
    '''
    if lang in args.train_from_scratch:
        model.add_adapter(lang, config=SeqBnConfig(reduction_factor=16))
    else:
        model.load_adapter(f"AdapterHub/bert-base-multilingual-cased-{lang}-wiki_pfeiffer")
    '''    

    model.add_adapter(TARGET_LANG, config=SeqBnConfig(reduction_factor=16))    

    # Set Active
    model.train_adapter([TARGET_LANG, "qa-task"])
    
    # Dataset & Dataloader
    # Select dictionary: 'en-th' or 'en-hi' etc
    
    train_ds = PAXDataset(SRC_LANG, "train", SOURCE_DICT[TARGET_LANG], tokenizer)
    test_ds =  PAXDataset(TARGET_LANG, "test", SOURCE_DICT[TARGET_LANG], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.to(device)
    # Epoch Loop
    best_accuracy = 0
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            #print(batch)
            #print(type(batch))
            #print(batch.keys())
            # prepare_train_features aligns character starts to token starts
            batch_code_switch
            batch["question"] = batch_code_switch(batch["question"], RATIO, current_dict, semantic_scorer, tokenizer, mode)
            #batch["question"] = [apply_codeswitch(txt, current_dict,mode ) for txt in batch["question"]]
            #print(batch)
            inputs = prepare_train_features(batch, tokenizer)
            inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items() if k in ['input_ids', 'attention_mask', 'start_positions', 'end_positions']}
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        current_acc,_ = evaluate_model(model,test_loader, TARGET_LANG,tokenizer,device)
        if current_acc > best_accuracy :
            best_accuracy = current_acc

    return best_accuracy    
        

        
if __name__ == "__main__":
    # Add your argparse logic here to pass into train_model(args)
    SEEDS = [i * 10 for i in range(1, 21)]
    #SEEDS = [10]#TODO
    results_list = []

    for s in SEEDS:
        res = {"seed": s}
        #["random", "selective_best"]
        #configs = ["random", "selective_best", 'english']
        #configs = ['selective_best']
        #configs = ['pos', 'random', 'english', 'selective_best']
        configs = ['english'] #TODO
        for m in configs:
            final_acc = train_model(s, m)
            res[m] = final_acc
            print(f"| Seed {s} | {m}: {final_acc:.2f}% |")
        results_list.append(res)

        if len(configs) == 1 and configs[0]=='english':
            pd.DataFrame(results_list).to_csv(f"paxqa_20seeds_english_{TARGET_LANG}.csv", index=False)

        elif len(configs) == 1 and configs[0]=='pos':
            pd.DataFrame(results_list).to_csv(f"paxqa_20seeds_pos_{TARGET_LANG}_{POS_TAGS}.csv", index=False)

        elif len(configs) == 1 and configs[0]=='random':
            pd.DataFrame(results_list).to_csv(f"paxxqa_20seeds_random_{TARGET_LANG}_{RATIO}.csv", index=False)
    
        
        else:    
            #pd.DataFrame(results_list).to_csv(f"paxqa_20seeds_{TARGET_LANG}_{THRESHOLD}_{TEMP}.csv", index=False)
            pd.DataFrame(results_list).to_csv(f"paxqa_20seeds_epochs_{TARGET_LANG}_{EPOCHS}.csv", index=False)
    