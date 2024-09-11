import torch
import torch.nn as nn
from transformers import T5Tokenizer
import argparse
from module import Solomon
import os
from utlis import TrainBatchify, EXPDataLoader, now_time, SEQDataLoader, ExpBatchify, SeqBatchify, TopNBatchify

#* #################################################################

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#* ORIGINAL PARAMETERS #############################################

model_version = "t5-small"
task_num = 3
prompt_num = 3
lr = 0.0005
epochs = 100
batch_size = 16 #64
log_interval = 200
endure_times = 5
exp_len = 20
negative_num = 99

#* LOAD_DATA #######################################################

data_name = "./data/toys/" #! wanted data
model_path = os.path.join(data_name, "model.pt")

print(now_time() + "Loading data...")
tokenizer = T5Tokenizer.from_pretrained(model_version)
exp_corpus = EXPDataLoader(data_name)
seq_corpus = SEQDataLoader(data_name)
nitem = len(seq_corpus.id2item)
all_iterator = TrainBatchify(exp_corpus.train, seq_corpus.user2items_positive, negative_num=negative_num, item_num=nitem, tokenizer=tokenizer, exp_len=exp_len, batch_size=batch_size)
exp_iterator = ExpBatchify(exp_corpus.train, tokenizer=tokenizer, exp_len=exp_len, batch_size=batch_size)
seq_iterator = SeqBatchify(seq_corpus.user2items_positive, tokenizer=tokenizer, batch_size=batch_size)
topn_iterator = TopNBatchify(seq_corpus.user2items_positive, seq_corpus.user2items_negative, negative_num=negative_num, item_num=nitem, tokenizer=tokenizer, batch_size=batch_size)

#* BUILD_MODEL #######################################################

model = Solomon.from_pretrained(model_version)
model.init_prompt(task_num=task_num, prompts_per_task=prompt_num, device=device)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

#* TRAIN #######################################################

model.load_state_dict(torch.load("./test/beauty/model.pth"))
model.eval()

def evaluate(iterator):
    
    model.eval()
    text_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            print("EVALUATION STEP")
            task, source, source_mask, whole_word, target = iterator.next_batch_valid()
            task = task.to(device)  # (batch_size,)
            source = source.to(device)  # (batch_size, seq_len)
            source_mask = source_mask.to(device)
            whole_word = whole_word.to(device)
            target = target.to(device)
            
            print("Before outputs")
            outputs = model(task, source, whole_word, source_mask, labels=target)
            print("After outputs")
            print("-" * 40)
            loss = outputs.loss
            batch_size = task.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size
            print("->>>>>>>>>>>> TEXT LOSS: ", text_loss)
            print("->>>>>>>>>>>> TOTAL SAMPLE: ", total_sample)
            if iterator.step == iterator.total_step:
                break
    return text_loss / total_sample

"""
[2024-09-09 15:12:36]validation
torch.Size([16, 11])
torch.Size([1, 11, 512])
torch.Size([16, 11])
torch.Size([0, 11, 512])
"""

exp_loss  = evaluate(exp_iterator)