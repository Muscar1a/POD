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


def train():
    
    model.train()
    text_loss = 0.
    total_sample = 0
    while True:
        task, source, source_mask, whole_word, target = all_iterator.next_batch()
        task = task.to(device)
        source = source.to(device)
        source_mask = source_mask.to(device)
        whole_word = whole_word.to(device)
        target = target.to(device)
        
        
        optimizer.zero_grad()
        outputs = model(task, source, whole_word, source_mask, labels=target)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        batch_size = task.size(0)
        text_loss += batch_size * loss.item()
        total_sample += batch_size
        
        if all_iterator.batch_index % log_interval == 0 or all_iterator.batch_index % all_iterator.batch_num == 0:
            cur_t_loss = text_loss / total_sample
            print(now_time() + "text loss {:4.4f} | {:5d}/{:5d} batches".format(cur_t_loss, all_iterator.batch_index, all_iterator.batch_num))
            text_loss = 0.
            total_sample = 0
        if all_iterator.batch_index % all_iterator.batch_num == 0:
            break
    print("Training done!")
# Ok

def evaluate(iterator):
    
    model.eval()
    text_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            task, source, source_mask, whole_word, target = iterator.next_batch_valid()
            task = task.to(device)  # (batch_size,)
            source = source.to(device)  # (batch_size, seq_len)
            source_mask = source_mask.to(device)
            whole_word = whole_word.to(device)
            target = target.to(device)
            outputs = model(task, source, whole_word, source_mask, labels=target)
            loss = outputs.loss

            batch_size = task.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size

            if iterator.step == iterator.total_step:
                break
    return text_loss / total_sample
                
                
if __name__ == '__main__':
    # with open(model_path, 'wb') as f:
    #     torch.save(model, f)
    
    print(now_time() + "Start training")
    best_val_loss = float('inf')
    endure_count = 0
    for epoch in range(1, epochs + 1):
        print(now_time() + "Epoch {}".format(epoch))
        train()
        print(now_time() + "Validation")
        exp_loss = evaluate(exp_iterator)
        print(now_time() + "Exp loss {:4.4f}".format(exp_loss))
        seq_loss = evaluate(seq_iterator)
        print(now_time() + "Seq loss {:4.4f}".format(seq_loss))
        topn_loss = evaluate(topn_iterator)
        print(now_time() + "TopN loss {:4.4f}".format(topn_loss))
        val_loss = exp_loss + seq_loss + topn_loss
        print(now_time() + "Total loss {:4.4f}".format(val_loss))
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(model_path, 'wb') as f:
                torch.save(model, f)
        else:
            endure_count += 1
            print(now_time() + "Endured {} time(s)".format(endure_count))
            if endure_count == endure_times:
                print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break