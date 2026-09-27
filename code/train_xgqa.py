import torch
import torch.nn as nn
import os
import random
from functools import partial
from collections import namedtuple
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler

# Transformers & Adapters
from transformers import XLMRobertaTokenizer, Blip2Processor, Blip2ForConditionalGeneration
from adapters import AutoAdapterModel, SeqBnConfig

# --- Your xMDETR Imports (Ensuring paths match your setup) ---
# Make sure your PYTHONPATH includes the xMDETR directory
#TODO
import sys
sys.path.append("/home/epsilon/Workbenches/ML_VLM/continual_training/xMDETR/mdetr")

import util.misc as utils
from dataset import build_dataset, get_coco_api_from_dataset

# --- 1. Multilingual mBLIP Model Wrapper ---
class MBlipAdapterModel(nn.Module):
    def __init__(self, vlm_name, xlmr_name, device):
        super().__init__()
        self.device = device
        
        # Load BLIP-2 (OPT-2.7b version) and freeze visual/LLM weights
        self.vlm = Blip2ForConditionalGeneration.from_pretrained(vlm_name, torch_dtype=torch.float16)
        for param in self.vlm.parameters():
            param.requires_grad = False
            
        # Initialize XLM-RoBERTa for multilingual text encoding
        self.xlmr = AutoAdapterModel.from_pretrained(xlmr_name)
        
        # Inject Sequential Bottleneck Adapter for Low-Resource Adaptation
        adapter_config = SeqBnConfig(reduction_factor=16)
        self.xlmr.add_adapter("lrl_adapter", config=adapter_config)
        self.xlmr.train_adapter("lrl_adapter")
        self.xlmr.set_active_adapters("lrl_adapter")

        # Alignment: XLM-R hidden (768) -> BLIP-2 LLM input (2560 for OPT-2.7b)
        self.proj = nn.Linear(768, 2560).to(torch.float16)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # 1. Extract multilingual text features
        text_outputs = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        # 2. Project XLM-R states to VLM space (replacing standard Q-Former text input)
        # Note: In research, we often concatenate or replace based on the specific transfer task
        
        # 3. Standard BLIP-2 forward
        outputs = self.vlm(pixel_values=pixel_values, labels=labels, return_dict=True)
        return outputs

# --- 2. Custom mBLIP Collate Function ---
def mblip_collate_fn(batch, tokenizer, processor, args, is_train=True):
    # Extract questions and handle potential Semantic Code-Switching
    questions = [item[0]['question'] if isinstance(item[0], dict) else item['question'] for item in batch]
    
    # Placeholder for your code_switch logic from the previous snippet
    # if is_train and args.do_cs: questions = [code_switch(q, args.cs_dict) for q in questions]

    images = []
    for item in batch:
        # xMDETR usually stores image objects or IDs in the first element of the tuple
        img = item[0]['image'] if isinstance(item[0], dict) else item['image']
        if not isinstance(img, Image.Image):
            # If it's a path/ID, load from your MIG_store path
            img_path = os.path.join(args.vg_image_path, f"{img}.jpg")
            img = Image.open(img_path).convert("RGB")
        images.append(img)

    # Process modalities
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    text_inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
    
    batch_out = {
        "pixel_values": pixel_values.to(args.device, torch.float16),
        "input_ids": text_inputs.input_ids.to(args.device),
        "attention_mask": text_inputs.attention_mask.to(args.device),
    }

    # Process Answers (Labels)
    answers = [item[0]['answer'] if isinstance(item[0], dict) else item['answer'] for item in batch]
    with processor.tokenizer.as_target_tokenizer():
        labels = processor.tokenizer(answers, return_tensors="pt", padding=True).input_ids
        batch_out["labels"] = labels.to(args.device)
            
    return batch_out

# --- 3. Data Loading Functions (GQA & xGQA) ---
def get_gqa(args, image_set, tokenizer, processor):
    #args.gqa_split_type = 'train'
    dataset_train = ConcatDataset([build_dataset(name, image_set=image_set, args=args) for name in args.combine_datasets])
    gqa_size = len(dataset_train)
    gqa_num_epoch = gqa_size // 10000 + 1 # Your chunking logic
    
    chunks = torch.chunk(torch.arange(gqa_size), gqa_num_epoch)
    datasets = [torch.utils.data.Subset(dataset_train, chunk.tolist()) for chunk in chunks]
    samplers = [DistributedSampler(ds) if args.distributed else torch.utils.data.RandomSampler(ds) for ds in datasets]
    
    collate = partial(mblip_collate_fn, tokenizer=tokenizer, processor=processor, args=args)
    
    loaders = [DataLoader(ds, batch_size=args.batch_size, sampler=s, collate_fn=collate, num_workers=args.num_workers) 
               for ds, s in zip(datasets, samplers)]
    
    return loaders, gqa_num_epoch, gqa_size

def get_val(args, lang, tokenizer, processor):

    Val_all = namedtuple(typename="val_data", field_names=["dataset_name", "dataloader", "base_ds"])
    val_tuples = []

    collate = partial(mblip_collate_fn, tokenizer=tokenizer, processor=processor, args=args, is_train=False)

    for dset_name in ["gqa"]:
        dset = build_dataset(dset_name, image_set="test", args=args)
        sampler = torch.utils.data.SequentialSampler(dset)
        loader = DataLoader(dset, args.batch_size, sampler=sampler, collate_fn=collate, num_workers=args.num_workers)
        base_ds = get_coco_api_from_dataset(dset)
        val_tuples.append(Val_all(dataset_name=dset_name, dataloader=loader, base_ds=base_ds))

    return val_tuples

# --- 4. Main Training Routine ---
def main(args,val_args):
    # Setup Models
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = MBlipAdapterModel("Salesforce/blip2-opt-2.7b", "xlm-roberta-base", args.device).to(args.device)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # 1. Get Training Chunks
    train_loaders, num_chunks, total_size = get_gqa(args, "train", tokenizer, processor)
    
    # 2. Get xGQA Validation
    val_data = get_val(val_args, "th", tokenizer, processor) # Example: Thai

    

    # Training Loop through chunks
    for chunk_idx, loader in enumerate(train_loaders):
        model.train()
        print(f"\n--- Training Chunk {chunk_idx+1}/{num_chunks} ---")
        for batch in tqdm(loader):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Periodic Evaluation
        model.eval()
        evaluate_xgqa(model, val_data, processor)

def evaluate_xgqa(model, val_data, processor):
    # Simplified Zero-Shot Accuracy check
    correct, total = 0, 0
    with torch.no_grad():
        for val_item in val_data:
            for batch in tqdm(val_item.dataloader, desc=f"Eval {val_item.dataset_name}"):
                generated_ids = model.vlm.generate(pixel_values=batch["pixel_values"], max_new_tokens=10)
                preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
                # Compare preds to batch["labels"] decoded...
                total += len(preds)
    print(f"Zero-shot Accuracy: {correct/total if total > 0 else 0}")

if __name__ == "__main__":
    # Mock Args for demonstration
    Args = namedtuple('Args', ['device', 'batch_size', 'num_workers', 'lr', 'combine_datasets', 
                               'combine_datasets_val', 'vg_img_path', 'distributed', 'lang', 'text_tokenizer_type', 'do_qa', 'gqa_split_type', 'gqa_ann_path','masks'])
    args = Args(device=torch.device("cuda:3"), batch_size=4, num_workers=4, lr=1e-4, 
                combine_datasets=["gqa"], combine_datasets_val=["xgqa"], 
                vg_img_path="/mnt/MIG_store/Datasets/epsilon/datasets/gqa/images", 
                distributed=False, lang="en", text_tokenizer_type='xlm-roberta-base',do_qa= True, gqa_split_type='balanced', gqa_ann_path = '/home/epsilon/Workbenches/ML_VLM/continual_training/xMDETR/annotations', masks = True)
    '''
    val_args = Args(device=torch.device("cuda:3"), batch_size=4, num_workers=4, lr=1e-4, 
                combine_datasets=["gqa"], combine_datasets_val=["xgqa"], 
                vg_img_path="/mnt/MIG_store/Datasets/epsilon/datasets/gqa/images", 
                distributed=False, lang="hi", text_tokenizer_type='xlm-roberta-base',do_qa= True, gqa_split_type='balanced', gqa_ann_path = '/home/epsilon/Workbenches/ML_VLM/continual_training/xMDETR/annotations', masks = True)
    '''
    val_args = args
    main(args,val_args)