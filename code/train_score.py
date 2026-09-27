import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import mean_squared_error, r2_score 
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
import os
from tqdm import tqdm 
import random

seed = 4
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- 1. Configuration ---
LANG1 = "en"
LANG2 = "hi" #TODO:
TASK_TYPE = "score_prediction"


#MODEL_NAME = "xlm-roberta-base"
MODEL_NAME = "sentence-transformers/LaBSE"
CSV_PATH = f"individual_word_cs_full_results_{LANG2}.csv" 
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
MAX_SAMPLES = 5000 #10000, 20000, 30000, ALL #TODO
TARGET_BUCKET = 'ALL' # ALL, HIGH, MEDIUM, LOW #TODO



# --- 2. Load and Preprocess Data (Refactored) ---

print(f"Loading tokenizer for {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_data(batch):
    """
    Tokenizes the code-switched sentences and copies the score.
    """
    # Tokenize the CODE-SWITCHED sentence as input
    tokenized_inputs = tokenizer(
        batch["code_switched_sentence"], 
        truncation=True,
        padding=False, # Collator will pad
        is_split_into_words=False,
    )
    
    # Copy the score as the label
    tokenized_inputs["labels_score"] = batch["similarity_score"]
    return tokenized_inputs

print(f"Loading and processing data from {CSV_PATH}")
try:
    df = pd.read_csv(CSV_PATH)
    if MAX_SAMPLES!='ALL' and (len(df) > MAX_SAMPLES):
        print(f"[TRAIN EXPERIMENT] Subsampling dataframe from {len(df):,} to {MAX_SAMPLES:,} rows...")
        #df = df.iloc[:MAX_SAMPLES]
        df = df.sample(n=MAX_SAMPLES, random_state=4).reset_index(drop=True)
    # Ensure 'code_switched_sentence' is a string for the tokenizer

    if TARGET_BUCKET != 'ALL':
        print(f"\n[EXPERIMENT] Isolating translation quality bucket: {TARGET_BUCKET}")
        
        if TARGET_BUCKET == 'HIGH':
            df = df[df['similarity_score'] >= 0.8].reset_index(drop=True)
        elif TARGET_BUCKET == 'MEDIUM':
            df = df[(df['similarity_score'] >= 0.5) & (df['similarity_score'] < 0.8)].reset_index(drop=True)
        elif TARGET_BUCKET == 'LOW':
            df = df[df['similarity_score'] < 0.5].reset_index(drop=True)
            
        print(f"[EXPERIMENT] Active rows in {TARGET_BUCKET} bucket: {len(df):,}\n")

    df['code_switched_sentence'] = df['code_switched_sentence'].astype(str)
except FileNotFoundError:
    print(f"ERROR: The file '{CSV_PATH}' was not found.")
    exit()
    
dataset = Dataset.from_pandas(df)
#dataset = dataset.filter(lambda example: example['similarity_score'] >= 0.5)


# -------------------  THE FIX IS HERE  -------------------
# You MUST remove the original 'similarity_score' column as well.


processed_dataset = dataset.map(
    preprocess_data, 
    batched=True, 
    remove_columns=['index', 'english_original', f'pure_{LANG2}_baseline', 'masked_indices', 'code_switched_sentence', 'similarity_score']
)
    # ---------------------------------------------------------

dataset_splits = processed_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset_splits["train"]
eval_dataset = dataset_splits["test"]

print(f"\nSample of processed data (train_dataset[0]):")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(train_dataset[0]['input_ids'])}")
print(f"Score Label: {train_dataset[0]['labels_score']}")


# --- 3. Custom Score Predictor Model (Refactored) ---

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
        print("\nLoRA model created:")
        self.base_model.print_trainable_parameters()

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Only one head is needed
        self.score_regressor = nn.Linear(config.hidden_size, 1)
        # Only one loss function is needed
        self.loss_fct_score = nn.MSELoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels_score=None,
        **kwargs
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get the [CLS] token's output
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)

        # Get score prediction
        score_logits = self.score_regressor(cls_output).squeeze(-1)

        total_loss = None
        
        if labels_score is not None:
            total_loss = self.loss_fct_score(score_logits, labels_score.float())
            
        return {
            "loss": total_loss,
            "score_logits": score_logits,
        }

# --- 4. Class Weight Calculation (REMOVED) ---


# --- 5. Training Setup (Refactored) ---
print("\n--- Setting up for Manual Training ---")

# Hyperparameters
NUM_EPOCHS = 100
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 5e-5
WARMUP_STEPS = 50

# Use new variables for the save path
BEST_MODEL_PATH = f"best_model_{LANG1}-{LANG2}_{TASK_TYPE}_{MAX_SAMPLES}_{TARGET_BUCKET}.pth" 

# Instantiate model
model = ScorePredictorModel(MODEL_NAME).to(device)

# Use standard DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

# Create DataLoaders
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=TRAIN_BATCH_SIZE
)
eval_dataloader = DataLoader(
    eval_dataset,
    collate_fn=data_collator,
    batch_size=EVAL_BATCH_SIZE
)

# Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Scheduler
num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=num_training_steps
)

# --- 6. Manual Training and Evaluation Loop (Refactored) ---

# We will maximize R-squared (higher is better)
best_score_r2 = -float("inf") 

for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
    
    # --- Training Phase ---
    model.train()
    total_train_loss = 0
    
    for batch in tqdm(train_dataloader, desc="Training"):
        # Only one label to pop
        labels_score = batch.pop("labels_score").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        outputs = model(
            **batch,
            labels_score=labels_score
        )
        
        loss = outputs["loss"]
        total_train_loss += loss.item()
        
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")

    # --- Evaluation Phase ---
    print("Running evaluation...")
    model.eval()
    
    all_true_scores = []
    all_pred_scores = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            labels_score = batch.pop("labels_score").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                **batch,
                labels_score=labels_score
            )
            
            score_logits = outputs["score_logits"]
            all_pred_scores.extend(score_logits.cpu().numpy())
            all_true_scores.extend(labels_score.cpu().numpy())
    
    # --- Calculate Metrics ---
    metrics = {}
    if not all_true_scores:
        metrics["score_mse"] = 0.0
        metrics["score_r2"] = 0.0
    else:
        metrics["score_mse"] = mean_squared_error(all_true_scores, all_pred_scores)
        metrics["score_r2"] = r2_score(all_true_scores, all_pred_scores)
    
    print("\nEvaluation Metrics:")
    print(pd.Series(metrics).to_string())

    # Save the best model based on R2 score
    current_metric = metrics["score_r2"] 
    if current_metric > best_score_r2:
        best_score_r2 = current_metric
        # The new filename will be used here
        print(f"New best R2 score: {current_metric:.4f}. Saving model to {BEST_MODEL_PATH}")
        torch.save(model.state_dict(), BEST_MODEL_PATH)

print("--- Training Finished ---")


# -----------------------------------------------------------------
# --- 7. Inference (Refactored) ---
# -----------------------------------------------------------------
print("\n--- Running Inference Example ---")

# 1. Re-instantiate the model structure
inference_model = ScorePredictorModel(MODEL_NAME).to(device)

# 2. Load the saved weights
try:
    # Use the same filename variable to load
    inference_model.load_state_dict(torch.load(BEST_MODEL_PATH))
    print(f"Loaded best model weights from {BEST_MODEL_PATH}")
except FileNotFoundError:
    print(f"ERROR: Model weights file not found at {BEST_MODEL_PATH}. Using untrained model for inference.")
except Exception as e:
    print(f"Error loading model weights: {e}. Using untrained model for inference.")

inference_model.eval() 

# Use a code-switched sentence as input
text = "Code बदलना is मुश्किल"
print(f"Input text: {text}")

inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = inference_model(**inputs) 

score_pred = outputs["score_logits"].item()
print(f"\nPredicted Cosine Score: {score_pred:.4f}")

print("\nScript Finished.")