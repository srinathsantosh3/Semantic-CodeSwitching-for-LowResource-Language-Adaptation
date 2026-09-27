import pandas as pd
from datasets import load_dataset
import torch
import random
import numpy as np
from tqdm import tqdm
import csv
import re
from sentence_transformers import SentenceTransformer, util
# Note: NllbTokenizer and AutoModelForSeq2SeqLM are not strictly necessary 
# for the current logic, but kept for context.
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, NllbTokenizer 
from typing import List
from collections import namedtuple, defaultdict
from nltk import word_tokenize
import traceback
import nltk 

# =========================================================================
# === CONFIGURATION AND SETUP =============================================
# =========================================================================

# --- Seeding for Reproducibility ---
seed = 4
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(torch.cuda.current_device())
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- GPU Configuration ---
# NOTE: Set your desired CUDA device ID here (e.g., cuda:0, cuda:1)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} {'(GPU)' if device.type == 'cuda' else '(CPU/Fallback)'}")

# =========================================================================
# === LANGUAGE CONFIGURATION (CHANGE THIS TO TEST OTHER LANGUAGES) ==========
TARGET_LANG = 'bn' # <<< SET TO BENGALI
SOURCE_LANG = 'en'
# =========================================================================

# Download NLTK data ('punkt') for word_tokenize
try:
    # Set quiet=True to suppress standard download messages
    nltk.download('punkt', quiet=True) 
    print("NLTK 'punkt' tokenizer data downloaded.")
except Exception as e:
    print(f"Could not download NLTK 'punkt' data. 'word_tokenize' might fail. Error: {e}")

# =========================================================================
# === DICTIONARY AND AUXILIARY FUNCTIONS ==================================
# =========================================================================

def load_code_switch(lang1, lang2, CODE_SWITCH_PATH):
    """Loads the bilingual dictionary for code-switching."""
    code_switch_dict = defaultdict(list)
    code_switch_dict_re = defaultdict(list)
    print("current code switch path: "+CODE_SWITCH_PATH)
    # Uses the configured source and target languages for the file path
    current_code_switch_path = CODE_SWITCH_PATH+"/"+lang1+"-"+lang2+".txt" 
    
    try:
        with open(current_code_switch_path, encoding='utf-8') as f:
            for line in f:
                # Basic tokenization for the dictionary entry
                parts = line.split()
                if len(parts) >= 2:
                    k, v = parts[0], " ".join(parts[1:])
                    # Normalize words for dictionary lookup
                    k, v = k.replace(".","").lower(), v.replace(".","").lower()
                    code_switch_dict[k].append(v)
                    code_switch_dict_re[v].append(k)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: The bilingual dictionary file was not found at:")
        print(f"{current_code_switch_path}")
        print("Please update the path in the script. The script will run, but no code-switching will occur.")
    except Exception as e:
        print(f"Error loading bilingual dictionary: {e}")
        
    print("Loading bilingual dictionary:", len(code_switch_dict))
    return code_switch_dict, code_switch_dict_re


# !! IMPORTANT: YOU MUST CHANGE THIS PATH TO YOUR EN-BN DICTIONARY FOLDER !!
if TARGET_LANG=='th':
    code_switch_dict, _ = load_code_switch(SOURCE_LANG, TARGET_LANG, "/home/epsilon/Workbenches/ML_VLM/continual_training/MTOP/dicts")
elif TARGET_LANG=='bn': # <<< BENGALI DICTIONARY PATH
    # *** CHANGE THIS LINE ***
    code_switch_dict, _ = load_code_switch(SOURCE_LANG, TARGET_LANG, "/home/epsilon/Workbenches/ML_VLM/lrl_adapt/muse")
else:
    code_switch_dict, _ = load_code_switch(SOURCE_LANG, TARGET_LANG, "/home/epsilon/Workbenches/ML_VLM/lrl_adapt/muse")

# --- Extract all target language words for wrong-word replacement ---
all_target_lang_words = []
for values in code_switch_dict.values():
    all_target_lang_words.extend(values)
all_target_lang_words = list(set(all_target_lang_words)) 
print(f"Total unique {TARGET_LANG} words loaded for wrong-word sampling: {len(all_target_lang_words):,}")
# -----------------------------------------------------------------------------


def code_switch_multilingual_pos(s, code_switch_dict, ratio, wrong_word_prob=0.15):
    """
    Performs code-switching, including intentional "wrong word" replacements 
    for robust data generation.
    """
    tokens = word_tokenize(s)
    if not tokens:
        return s, []
        
    mask = [0] * len(tokens)
    codeswitch_target_count = max(1, int(ratio * len(tokens))) 
    words_replacement_pos = 0
    
    try:
        sample_size = min(codeswitch_target_count, len(tokens))
        
        if sample_size > 0:
            # Randomly sample indices for potential switching
            for idx in random.sample(list(range(len(tokens))), k=sample_size):
                # Ensure original case is preserved (e.g., proper nouns), but dictionary lookup is lowercase
                original_token = tokens[idx]
                target = original_token.lower()
                
                # 1. Check if the word is translatable and we haven't hit the target count
                if (words_replacement_pos < codeswitch_target_count) and (target in code_switch_dict):
                    
                    # 2. Decide between Correct Translation or Wrong Word
                    if random.random() < wrong_word_prob:
                        # --- WRONG WORD REPLACEMENT (Simulates semantic error) ---
                        if all_target_lang_words:
                            # Replace with a random word from the whole target language list
                            tokens[idx] = random.choice(all_target_lang_words) 
                        else:
                            # Fallback to correct word if the list is empty
                            tokens[idx] = code_switch_dict.get(target)[0]
                            
                    else:
                        # --- CORRECT TRANSLATION (Simulates fluent code-switch) ---
                        tokens[idx] = code_switch_dict.get(target)[0]
                        
                    # 3. Mark and Increment
                    words_replacement_pos += 1
                    mask[idx] = 1 
                else:
                    continue
        
        s = ' '.join(tokens)
        return s, mask
            
    except Exception as e:
        print("In exception during 'code_switch_multilingual_pos':")
        print(f"Error: {e}")
        traceback.print_exc()
        s = ' '.join(tokens)
        return s, mask 


def contains_english_word(text):
    """Checks if a string contains any word composed purely of Latin (English) characters."""
    if not isinstance(text, str):
        return True
    return bool(re.search(r'\b[a-zA-Z]+\b', text))

def filter_mixed_rows(example, target_lang_field):
    """Filter function to keep only rows where the target language is pure."""
    return not contains_english_word(example[target_lang_field])


# =========================================================================
# === DATASET LOADING AND FILTERING =======================================
# =========================================================================

# --- 1. Load and Filter Dataset (WITH BENGALI INTEGRATION) ---

try:
    print(f"Loading {SOURCE_LANG}-{TARGET_LANG} dataset...")
    
    # --- Integration of csebuetnlp/BanglaNMT ---
    if TARGET_LANG == 'bn' and SOURCE_LANG == 'en': 
        dataset_name = "csebuetnlp/BanglaNMT"
        dataset_config = None # Use default config
        english_lang_field = 'english_text'
        target_lang_field = 'bengali_text'
        print(f"Using **{dataset_name}** for EN-BN.")
        
    # --- Integration of airesearch/scb_mt_enth_2020 (Original THAI block) ---
    elif TARGET_LANG == 'th' and SOURCE_LANG == 'en':
        dataset_name = "airesearch/scb_mt_enth_2020"
        dataset_config = 'enth' 
        english_lang_field = 'en'
        target_lang_field = 'th'
        print(f"Using **{dataset_name}** for EN-TH.")
        
    else:
        # Fallback to a general parallel corpus (WikiMatrix, etc.)
        dataset_name = "sentence-transformers/parallel-sentences-wikimatrix"
        dataset_config = f"{SOURCE_LANG}-{TARGET_LANG}" 
        english_lang_field = 'english'
        target_lang_field = 'non_english'
        print(f"Using **{dataset_name}** with config {dataset_config} for EN-{TARGET_LANG.upper()}.")

    # Load the dataset
    dataset = load_dataset(dataset_name, dataset_config, split="train") 
    TOTAL_SAMPLES_BEFORE = len(dataset)
    
    
    # --- FIX 1: Un-nest the language columns for THAI (scb_mt_enth_2020) ---
    if TARGET_LANG == 'th' and 'translation' in dataset.features:
        print("Extracting nested 'en' and 'th' columns from 'translation'...")
        dataset = dataset.map(lambda example: {
            'en': example['translation']['en'],
            'th': example['translation']['th']
        }, remove_columns=['translation'])
        
        
    # --- FIX 2: Correct Column Renaming/Selection for ALL TARGETS ---
    # Unify all dataset column names to 'english' and 'non_english'
    
    # 2a. Handle THAI renaming 
    if TARGET_LANG == 'th':
        # Thai columns are now 'en' and 'th' after mapping.
        dataset = dataset.rename_column('en', 'english').rename_column('th', 'non_english')
        
    # 2b. Handle BENGALI renaming 
    elif TARGET_LANG == 'bn':
        # BanglaNMT uses 'english_text' and 'bengali_text'
        #dataset = dataset.rename_column(english_lang_field, 'english').rename_column(target_lang_field, 'non_english')
        dataset = dataset.rename_column('en', 'english').rename_column('bn', 'non_english')
    # After renaming, the target field for filtering and processing is always 'non_english'
    target_lang_field = 'non_english' 
    english_lang_field = 'english'
    
    
    # Filtering applied only if the target language is not one known to be highly mixed 
    if TARGET_LANG not in ['id', 'de', 'fr', 'es']: 
        print(f"Filtering dataset to remove pairs with English words in the '{TARGET_LANG}' column...")
        dataset = dataset.filter(lambda example: filter_mixed_rows(example, target_lang_field))
        
    TOTAL_SAMPLES_AFTER = len(dataset)
    removed_count = TOTAL_SAMPLES_BEFORE - TOTAL_SAMPLES_AFTER
    
    print(f"Dataset loaded and filtered successfully.")
    print(f"Dataset Features: {dataset.features}")
    print(f"Total rows loaded: {TOTAL_SAMPLES_BEFORE:,} | Rows remaining after filter: {TOTAL_SAMPLES_AFTER:,} | Rows removed: {removed_count:,}")
        
except Exception as e:
    print(f"Error loading or filtering dataset: {e}")
    # exit() 

# ----------------------------------------------------------------------

# =========================================================================
# === MODEL LOADING AND PROCESSING ========================================
# =========================================================================

# --- 2. Load Models for Embedding and Translation ---

print("\nLoading models for Embedding (LaBSE) and Translation...")

# 2a. Embedding Model (for similarity scoring)
try:
    # LaBSE is a good multilingual choice
    model_name = 'sentence-transformers/LaBSE' 
    similarity_model = SentenceTransformer(model_name).to(device)
    print(f"{model_name} loaded.")
except Exception as e:
    print(f"Error loading Embedding model: {e}")
    # exit()


# --- 3. Individual Word Translation-Based Code-Switching Function (WRAPPER - Unchanged) ---

def translate_individual_words_nllb(en_sentence: str, cs_percentage: float = 0.95):
    """
    Wrapper for the code-switching function.
    """
    try:
        # Use a random ratio to ensure variability in switching density
        cs_percentage = round(random.uniform(0.000001, 0.999999), 1) 
        
        # Calls the updated function with wrong_word_prob=0.15 (default)
        cs_sentence, cs_mask = code_switch_multilingual_pos(en_sentence, code_switch_dict, cs_percentage)
        return cs_sentence, cs_mask
    
    except Exception as e:
        print(f"Error during code-switching for sentence: '{en_sentence}'")
        print(f"Error: {e}")
        return en_sentence, []


# --- 4. Processing and Similarity Calculation (SLIGHTLY MODIFIED) ---
def process_full_dataset(dataset, model, current_device, batch_size=64):
    """
    Applies individual word translation-based code-switching, calculates similarity, 
    and collects data for ALL processed rows.
    """
    all_similarity_scores = []
    full_results = [] 
    
    if len(dataset) == 0:
        print("Dataset is empty after filtering. Skipping processing.")
        return [], []
        
    print(f"\nStarting processing of {len(dataset):,} examples with batch size {batch_size}...")

    model.eval() 

    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing Batches"):
        
        end_index = min(i + batch_size, len(dataset))
        batch = dataset[i:end_index]
        
        # NOTE: 'english' and 'non_english' are now the unified column names
        english_sentences = batch['english']   
        pure_target_sentences = batch['non_english']   
        
        # 1. Generate Individual Word Translation-Based Code-Switched Sentences
        code_switched_sentences = []
        code_switched_masks = []
        
        for en_sent in english_sentences:
            cs_sentence, cs_mask = translate_individual_words_nllb(en_sent, cs_percentage=0.50)
            code_switched_sentences.append(cs_sentence)
            code_switched_masks.append(cs_mask)

        # 2. Encode all sentences in bulk (on GPU)
        try:
            with torch.no_grad():
                # Encodes pure_target_sentences (e.g., pure_bn_sentences)
                target_embeddings = model.encode(pure_target_sentences, convert_to_tensor=True, show_progress_bar=False, device=current_device.type)
                cs_embeddings = model.encode(code_switched_sentences, convert_to_tensor=True, show_progress_bar=False, device=current_device.type)

                cosine_scores = util.cos_sim(cs_embeddings, target_embeddings).diag()
            
            # 4. Store Results and collect ALL data
            batch_scores = [score.item() for score in cosine_scores.cpu()]
            all_similarity_scores.extend(batch_scores)
            
            start_index = i
            for k in range(len(batch_scores)):
                mask_list = code_switched_masks[k]
                mask_string = " ".join(map(str, mask_list))
                
                full_results.append({
                    'index': start_index + k,
                    f'english_original': english_sentences[k],
                    f'pure_{TARGET_LANG}_baseline': pure_target_sentences[k],
                    'code_switched_sentence': code_switched_sentences[k],
                    'masked_indices': mask_string,
                    'similarity_score': batch_scores[k]
                })
        
        except Exception as e:
            print(f"\nError during batch starting at index {i}: {e}")
            print(f"Skipping this batch.")
            continue
        
    return all_similarity_scores, full_results 

# ----------------------------------------------------------------------

# --- 5. Execution ---
if 'dataset' in locals() and dataset is not None and len(dataset) > 0:
    # NOTE: Set your preferred batch size here
    similarity_scores, full_data = process_full_dataset(dataset, similarity_model, current_device=device, batch_size=64) 
else:
    print("Dataset not loaded or is empty. Skipping processing.")
    similarity_scores, full_data = [], []


# --- 6. Analysis and Saving Results (Unchanged) ---
if similarity_scores:
    scores_array = np.array(similarity_scores)
    mean_score = scores_array.mean()
    std_score = scores_array.std()
    max_score = scores_array.max()
    min_score = scores_array.min()

    print("\n" + "="*50)
    print(f"   STATISTICAL ANALYSIS OF INDIVIDUAL WORD CS (EN-{TARGET_LANG.upper()})")
    print("="*50)
    print(f"Total Samples Analyzed:                     {len(similarity_scores):,}")
    print(f"Target Language:                            {TARGET_LANG.upper()}")
    print("-" * 50)
    print(f"Mean (Average) Cosine Similarity:           {mean_score:.4f}")
    print(f"Standard Deviation (SD) of Scores:          {std_score:.4f}")
    print(f"Maximum Similarity Score:                   {max_score:.4f}")
    print(f"Minimum Similarity Score:                   {min_score:.4f}")
    print("="*50)
    
    stats_file = f'individual_word_cs_full_stats_filtered_{TARGET_LANG}.csv'
    # Saving stats file logic remains here
    
    full_data_file = f'individual_word_cs_full_results_{TARGET_LANG}.csv'
    
    if full_data:
        print(f"\nAttempting to save {len(full_data):,} rows to CSV. This may take a while.")
        
        try:
            keys = full_data[0].keys()
        except IndexError:
            print("full_data list is empty. Cannot save CSV or show results.")
            keys = None

        if keys:
            try:
                with open(full_data_file, 'w', newline='', encoding='utf-8') as f:
                    dict_writer = csv.DictWriter(f, keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(full_data)
                
                full_df = pd.DataFrame(full_data)
                
                # --- START: TOP 10 MAX/MIN SCORES (Analysis) ---
                
                # Get Top 10 Max
                top_10_max = full_df.nlargest(10, 'similarity_score')
                print("\n" + "*"*60)
                print(f"       TOP 10 MOST SIMILAR SENTENCE PAIRS (EN-{TARGET_LANG.upper()})")
                print(f"       (Individual Word CS vs. Pure {TARGET_LANG.upper()} Baseline)")
                print("*"*60)
                for i, (idx, max_row) in enumerate(top_10_max.iterrows()):
                    print(f"\n--- Top {i+1} (Score: {max_row['similarity_score']:.4f}) ---")
                    print(f"Index: {max_row['index']}")
                    print(f"Code-Switched: {max_row['code_switched_sentence']}")
                    print(f"Masked Indices: {max_row['masked_indices']}")
                    print(f"Pure {TARGET_LANG.upper()} Baseline: {max_row[f'pure_{TARGET_LANG}_baseline']}")
                    print(f"Original {SOURCE_LANG}: {max_row[f'english_original']}")
                print("*"*60)

                # Get Top 10 Min
                top_10_min = full_df.nsmallest(10, 'similarity_score')
                print("\n" + "*"*60)
                print(f"       TOP 10 LEAST SIMILAR SENTENCE PAIRS (EN-{TARGET_LANG.upper()})")
                print(f"       (Individual Word CS vs. Pure {TARGET_LANG.upper()} Baseline)")
                print("*"*60)
                for i, (idx, min_row) in enumerate(top_10_min.iterrows()):
                    print(f"\n--- Bottom {i+1} (Score: {min_row['similarity_score']:.4f}) ---")
                    print(f"Index: {min_row['index']}")
                    print(f"Code-Switched: {min_row['code_switched_sentence']}")
                    print(f"Masked Indices: {min_row['masked_indices']}")
                    print(f"Pure {TARGET_LANG.upper()} Baseline: {min_row[f'pure_{TARGET_LANG}_baseline']}")
                    print(f"Original {SOURCE_LANG}: {min_row[f'english_original']}")
                print("*"*60)
                # --- END: TOP 10 MAX/MIN SCORES ---
                
            except Exception as e:
                print(f"\nCRITICAL ERROR during saving or final analysis: {e}")
                traceback.print_exc()
                print(f"The data may still be available in the {full_data_file} file.")
            
    else:
        print("No similarity scores or data were collected. The dataset may have been filtered to zero rows.")
else:
    print("No similarity scores were collected. The dataset may have been empty or an error occurred.")


# --- Final Model Files Check ---
print("\nModel saving finished. Check your file system for the output.")
print("\nScript finished.")