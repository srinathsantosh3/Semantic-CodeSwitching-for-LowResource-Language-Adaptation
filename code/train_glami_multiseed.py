import torch
import torch.nn as nn
import numpy as np
import random
import os
import gc
import pandas as pd
import nltk
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    ViTModel, 
    ViTImageProcessor,
)
from torch.optim import AdamW 
from peft import get_peft_model, LoraConfig
from sklearn.metrics import accuracy_score, f1_score
from nltk.tokenize import word_tokenize
from collections import defaultdict
from scipy import stats

# Ensure resources
nltk.download('punkt')

# --- 1. Global Configuration ---
PRECOMPUTE_MODE = False  # Step 1: Set to True. Step 2: Set to False for training.

TARGET_LANG = "de" #TODO
BASE_LANG = "de" #TODO
TEXT_MODEL = "xlm-roberta-base"
VISION_MODEL = "google/vit-base-patch16-224-in21k"
SCORER_MODEL = "sentence-transformers/LaBSE"
DEVICE_ID = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
DEVICE = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

IMG_ROOT = "/mnt/MIG_store/Datasets/epsilon/datasets/GLAMI/GLAMI-1M-dataset/images" 
ORIGINAL_TRAIN_CSV = "/home/epsilon/Workbenches/ML_VLM/lrl_adapt/scripts/glami_processed_labse.csv"
#AUGMENTED_TRAIN_CSV = f"/home/epsilon/Workbenches/ML_VLM/lrl_adapt/scripts/glami_train_{TARGET_LANG}_augmented.csv"
AUGMENTED_TRAIN_CSV = f"/home/epsilon/Workbenches/ML_VLM/lrl_adapt/scripts/glami_de_train.csv" #TODO
if TARGET_LANG in ['hi', 'th']:
    TEST_CSV = "/home/epsilon/Workbenches/ML_VLM/lrl_adapt/scripts/glami_hi_th_combined_test.csv"
elif TARGET_LANG =="de":
    TEST_CSV = "/home/epsilon/Workbenches/ML_VLM/lrl_adapt/scripts/glami_de_combined_test.csv"
else: 
    TEST_CSV = "/mnt/MIG_store/Datasets/epsilon/datasets/GLAMI/GLAMI-1M-dataset/GLAMI-1M-test.csv"
MUSE_PATH = "/home/epsilon/Workbenches/ML_VLM/lrl_adapt/muse" 

#THRESHOLD_DICT = {'hi':0.63, 'th': 0.52, 'de': 0.62, 'es': 0.64}
THRESHOLD_DICT = {'hi':0.55, 'th': 0.52, 'de': 0.62, 'es': 0.55}
#th ok, de ok, Phase:2, de F1 is NaN
#es and hi reduced to 0.55 => creating selective test: Phase1
MAX_LEN = 128
EPOCHS = 3 #TODO
BATCH_SIZE = 32
RATIO = 0.5       
#THRESHOLD = 0.5
THRESHOLD = THRESHOLD_DICT[TARGET_LANG] 
#0.63: Hindi 
#0.52: Thai
#0.62: German
#0.54: Spanish
TEMP = 0.3       
POS_TAGS = ['NOUN']
#ADJ, ADV,INTJ, NOUN,PROPN, VERB

# The 27 canonical GLAMI benchmark categories
GLAMI_27_CATEGORIES = [
    "dresses", "t-shirts", "sneakers", "sweatshirts", "bags", "pants", 
    "jackets", "sweaters", "sandals", "underwear", "skirts", "shirts", 
    "shorts", "blouses", "boots", "coats", "tops", "jewelry", 
    "swimwear", "accessories", "watches", "pajamas", "nightgowns", 
    "overalls", "leggings", "suits", "socks"
]

def map_slug_to_27(slug):
    """
    Maps a fine-grained slug (e.g., 'mens-t-shirts-and-tank-tops') 
    to one of the 27 benchmark categories.
    """
    slug = str(slug).lower()
    # Check for keywords in order of specificity
    for target in GLAMI_27_CATEGORIES:
        if target in slug:
            return target
    return "other" # Fallback for categories outside the 27

'''
# --- 2. Label Mapping Setup ---
def get_label_map():
    train_path = AUGMENTED_TRAIN_CSV if os.path.exists(AUGMENTED_TRAIN_CSV) else ORIGINAL_TRAIN_CSV
    full_train = pd.read_csv(train_path)
    full_test = pd.read_csv(TEST_CSV)
    #all_categories = sorted(pd.concat([full_train['category'], full_test['category']]).unique()) #TODO
    all_categories = sorted(pd.concat([full_train['category_name'], full_test['category_name']]).unique())
    print(f"Total Unique Classes: {len(all_categories)}, Expected: 27")
    return {raw_id: i for i, raw_id in enumerate(all_categories)}

''' 

def get_label_map():
    # We define the map based on the 27 targets, not the CSV contents
    # This ensures consistency across train/test even if a class is missing in one
    categories = sorted(GLAMI_27_CATEGORIES + ["other"])
    print(f"Total Unique Classes: {len(categories)}, Expected: 27 (+1 other)")
    return {cat_name: i for i, cat_name in enumerate(categories)}

label_map = get_label_map()
NUM_LABELS = len(label_map)

# --- 3. CS Logic & Model Classes ---
def load_cs_resources(l1, l2, path):
    cs_dict = defaultdict(list)
    file_path = f"{path}/{l1}-{l2}.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2: cs_dict[parts[0].lower()].append(parts[1].lower())
    return cs_dict

def batch_code_switch(texts, ratio, cs_dict, scorer, tokenizer, mode, precomputed_texts=None):
    if mode == 'selective_best' and precomputed_texts is not None:
        return [str(t) if pd.notna(t) else "" for t in precomputed_texts]

    if mode == 'none' or not cs_dict: return texts
    
    # Defensive check: Convert all inputs to strings
    texts = [str(t) if pd.notna(t) else "" for t in texts]

    if mode == 'english':
        return texts
    
    if mode == "random":
        final_texts = []
        for text in texts:
            tokens = word_tokenize(text)
            swappable = [i for i, w in enumerate(tokens) if w.lower() in cs_dict]
            if not swappable: final_texts.append(text); continue
            k = min(max(1, int(ratio * len(tokens))), len(swappable))
            for idx in random.sample(swappable, k):
                tokens[idx] = cs_dict[tokens[idx].lower()][0]
            final_texts.append(' '.join(tokens))
        return final_texts
    
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

    if mode == "selective_best":
        all_candidates, metadata = [], []
        for text in texts:
            tokens = word_tokenize(text)
            swp = [i for i, w in enumerate(tokens) if w.lower() in cs_dict]
            if not swp: metadata.append(None); continue
            k = min(max(1, int(ratio * len(tokens))), len(swp))
            cands = []
            for idx in swp:
                trans = cs_dict[tokens[idx].lower()][0]
                #print(f"trans:{trans}")
                tmp = list(tokens); tmp[idx] = trans
                cands.append({"idx": idx, "trans": trans, "sent": ' '.join(tmp)})
                all_candidates.append(' '.join(tmp))
            metadata.append({"tokens": tokens, "k": k, "candidates": cands})

        if not all_candidates: return texts

        scorer.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(all_candidates), 128):
                inputs = tokenizer(all_candidates[i : i + 128], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                scores = torch.sigmoid(scorer(inputs.input_ids, inputs.attention_mask)).cpu().tolist()
                #print(f"scores: {scores}")
                all_scores.extend(scores)

        final_texts, score_ptr = [], 0
        for m in metadata:
            if m is None: final_texts.append(""); continue
            cur_cands = m["candidates"]
            cur_scores = all_scores[score_ptr : score_ptr + len(cur_cands)]
            score_ptr += len(cur_cands)
            
            valid = [c for i, c in enumerate(cur_cands) if cur_scores[i] >= THRESHOLD]
            if not valid: 
                final_texts.append(' '.join(m["tokens"])); continue
            
            valid_scores = np.array([cur_scores[cur_cands.index(v)] for v in valid])
            probs = np.exp(valid_scores / TEMP) / np.sum(np.exp(valid_scores / TEMP))
            chosen = np.random.choice(len(valid), size=min(m["k"], len(valid)), replace=False, p=probs)
            res_tokens = list(m["tokens"])
            for c_idx in chosen: res_tokens[valid[c_idx]["idx"]] = valid[c_idx]["trans"]
            final_texts.append(' '.join(res_tokens))
            #print(f"final_texts: {final_texts}")

        return final_texts

class ScorePredictorModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        self.base = get_peft_model(self.base, LoraConfig(r=8, target_modules=["query", "value"]))
        self.regressor = nn.Linear(self.base.config.hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids, attention_mask).last_hidden_state[:, 0]
        return self.regressor(out).squeeze(-1)

class MultimodalClassifier(nn.Module):
    def __init__(self, text_model, vis_model, num_labels):
        super().__init__()
        self.text_enc = get_peft_model(AutoModel.from_pretrained(text_model), LoraConfig(r=8, target_modules=["query", "value"]))
        self.vis_enc = get_peft_model(ViTModel.from_pretrained(vis_model), LoraConfig(r=8, target_modules=["query", "value"]))
        self.classifier = nn.Linear(self.text_enc.config.hidden_size + self.vis_enc.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask, pixel_values):
        t_feat = self.text_enc(input_ids, attention_mask).last_hidden_state[:, 0]
        v_feat = self.vis_enc(pixel_values=pixel_values).last_hidden_state[:, 0]
        return self.classifier(torch.cat((t_feat, v_feat), dim=1))

# --- 4. Dataset Loader ---

'''
class GlamiLocalDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_root, lang_code, processor, label_map):
        df = pd.read_csv(csv_path)
        lang_col = 'geo' if 'geo' in df.columns else 'language'
        self.lang_code = lang_code
        # Filtering: train on EN (using 'sk' as per your source), test on TARGET_LANG
        self.df = df[df[lang_col] == ('sk' if lang_code == 'en' else lang_code)].copy()
        self.img_root = img_root
        self.processor = processor
        self.label_map = label_map

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        #label_idx = self.label_map[row['category']] #TODO
        label_idx = self.label_map[row['category_name']] 
        img_path = os.path.join(self.img_root, f"{row['image_id']}.jpg")

        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        except:
            pixel_values = torch.zeros(3, 224, 224)

        # Handle NaNs in text columns during runtime
        name = str(row['name']) if pd.notna(row['name']) else ""
        desc = str(row['description']) if pd.notna(row.get('description', "")) else ""
        if self.lang_code=='en':
            trans = str(row['eng_trans']) if pd.notna(row.get('eng_trans', "")) else ""

        item = {
            "pixel_values": pixel_values, 
            "text": f"{name} {trans}" if self.lang_code=='en' else f"{name} {desc}", 
            "label": torch.tensor(label_idx, dtype=torch.long), 
            "weight": torch.tensor(1.0 if row['label_source'] == 'human' else 0.5)
        }
        if 'text_selective_best' in row:
            item['text_selective_best'] = str(row['text_selective_best']) if pd.notna(row['text_selective_best']) else ""
        return item
'''


class GlamiLocalDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_root, lang_code, processor, label_map):
        df = pd.read_csv(csv_path)
        lang_col = 'geo' if 'geo' in df.columns else 'language'
        self.lang_code = lang_code
        self.df = df[df[lang_col] == ('sk' if lang_code == 'en' else lang_code)].copy()
        self.img_root = img_root
        self.processor = processor
        self.label_map = label_map

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # MAP SLUG TO HIGH-LEVEL CLASS
        fine_slug = row['category_name']
        high_level_cat = map_slug_to_27(fine_slug)
        label_idx = self.label_map[high_level_cat]

        # Image Loading
        img_path = os.path.join(self.img_root, f"{row['image_id']}.jpg")
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        except:
            pixel_values = torch.zeros(3, 224, 224)

        # Text Handling (preserving your existing logic)
        name = str(row['name']) if pd.notna(row['name']) else ""
        desc = str(row['description']) if pd.notna(row.get('description', "")) else ""
        trans = str(row.get('eng_trans', "")) if self.lang_code == 'en' and pd.notna(row.get('eng_trans')) else ""

        item = {
            "pixel_values": pixel_values, 
            "text": f"{name} {trans}" if self.lang_code=='en' else f"{name} {desc}", 
            "label": torch.tensor(label_idx, dtype=torch.long), 
            "weight": torch.tensor(1.0 if row['label_source'] == 'human' else 0.5)
        }
        
        if 'text_selective_best' in row:
            item['text_selective_best'] = str(row['text_selective_best']) if pd.notna(row['text_selective_best']) else ""
            
        return item



# --- 5. Precompute Logic ---
def run_precomputation(): #helps to create code switch data 
    print("--- STARTING PRECOMPUTATION ---")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    scorer = ScorePredictorModel(SCORER_MODEL).to(DEVICE)
    cs_dict = load_cs_resources(BASE_LANG, TARGET_LANG, MUSE_PATH)
    
    df = pd.read_csv(ORIGINAL_TRAIN_CSV)
    lang_col = 'geo' if 'geo' in df.columns else 'language'
    train_mask = df[lang_col] == 'sk'
    
    # SAFE STRING EXTRACTION
    names = df.loc[train_mask, 'name'].fillna('').astype(str)
    trans = df.loc[train_mask, 'eng_trans'].fillna('').astype(str)
    train_texts = (names + " " + trans).tolist()
    
    augmented_texts = []
    for i in tqdm(range(0, len(train_texts), BATCH_SIZE)):
        batch = train_texts[i : i + BATCH_SIZE]
        aug = batch_code_switch(batch, RATIO, cs_dict, scorer, tokenizer, mode="selective_best")
        augmented_texts.extend(aug)
    
    df.loc[train_mask, 'text_selective_best'] = augmented_texts
    df.to_csv(AUGMENTED_TRAIN_CSV, index=False)
    print(f"Success! Augmented CSV saved to {AUGMENTED_TRAIN_CSV}")

# --- 6. Main Experiment ---
def run_glami_experiment(seed, mode, label_map):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    processor = ViTImageProcessor.from_pretrained(VISION_MODEL)
    cs_dict = load_cs_resources(BASE_LANG, TARGET_LANG, MUSE_PATH)
    
    model = MultimodalClassifier(TEXT_MODEL, VISION_MODEL, NUM_LABELS).to(DEVICE)
    train_ds = GlamiLocalDataset(AUGMENTED_TRAIN_CSV, IMG_ROOT, BASE_LANG, processor, label_map)
    print(f"length of dataset: {len(train_ds)}")
    test_ds = GlamiLocalDataset(TEST_CSV, IMG_ROOT, TARGET_LANG, processor, label_map)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc=f"Seed {seed} | {mode}"):
            pre_tx = []
            if mode!= 'english':
                pre_tx = batch.get("text_selective_best", None)
            aug_texts = batch_code_switch(batch["text"], RATIO, cs_dict, None, tokenizer, mode, precomputed_texts=pre_tx)
            t_in = tokenizer(aug_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            logits = model(t_in.input_ids, t_in.attention_mask, batch["pixel_values"].to(DEVICE))
            loss = (criterion(logits, batch["label"].to(DEVICE)) * batch["weight"].to(DEVICE)).mean()
            loss.backward(); optimizer.step(); optimizer.zero_grad()

    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in test_loader:
            t_in = tokenizer(batch["text"], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            logits = model(t_in.input_ids, t_in.attention_mask, batch["pixel_values"].to(DEVICE))
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            golds.extend(batch["label"].numpy())
            #print(f"preds: {preds}") 
            #print(f"golds: {golds}")           

    return f1_score(golds, preds, average='macro') * 100

if __name__ == "__main__":
    if PRECOMPUTE_MODE: #helps to create code switch data 
        run_precomputation()
    else:
        SEEDS = [i * 10 for i in range(1, 21)]
        #MODES = ['english','random', 'selective_best', 'pos']
        MODES = ['english']
        results = []
        for s in SEEDS:
            res = {"seed": s}
            for m in MODES:
                score = run_glami_experiment(s, m, label_map)
                res[m] = score
                print(f"Seed {s} | {m} | F1: {score:.2f}%")
            results.append(res)
            if len(MODES)==1 and MODES[0]=='pos':
                pd.DataFrame(results).to_csv(f"arr_results_pos_{'_'.join(POS_TAGS)}_{TARGET_LANG}.csv", index=False)
            elif len(MODES)==1 and MODES[0]=='english':
                pd.DataFrame(results).to_csv(f"arr_results_pos_{BASE_LANG}_{TARGET_LANG}.csv", index=False)
            else:
                pd.DataFrame(results).to_csv(f"arr_results_{TARGET_LANG}.csv", index=False)