import os
import torch 
import random 
from transformers import T5Tokenizer
from utlis import SEQDataLoader, SeqBatchify, now_time, evaluate_ndcg, evaluate_hr


data_dir = "./data/beauty/"
model_version = "t5-small"
batch_size = 32
checkpoint = "./checkpoint/beauty/"
num_beams = 20
top_n = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(now_time() + "Loading data...")
tokenizer = T5Tokenizer.from_pretrained(model_version)
seq_corpus = SEQDataLoader(data_dir)
nitem = len(seq_corpus.id2item)
seq_iterator = SeqBatchify(seq_corpus.user2items_positive, tokenizer=tokenizer, batch_size=batch_size)

#* TEST MODEL #################################################################

model_path = os.path.join(checkpoint, "model.pt")
with open(model_path, "rb") as f:
    model = torch.load(f).to(device)
    
def generate():
    model.eval()
    idss_predict = []
    with torch.no_grad():
        print("Hello")
        while True:
            print("In loop")
            task, source, source_mask, whole_word, _ = seq_iterator.next_batch()
            task = task.to(device)
            source = source.to(device)
            source_mask = source_mask.to(device)
            whole_word = whole_word.to(device)
            
            beam_outputs = model.my_beam_search(task, source, whole_word, source_mask, 
                                                num_beams=num_beams, 
                                                num_return_sequences=top_n
                                                )
            print(task)
            break 
            

idss_predicted = generate()