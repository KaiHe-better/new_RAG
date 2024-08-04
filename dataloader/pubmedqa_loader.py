import json
import numpy as np
import pandas as pd
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import os


class PubmedQA(Dataset):

    def __init__(self, args, data_file, tokenizer):
        self.args = args
        self.LLM_tokenizer = tokenizer
        
        with open(data_file, 'r') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        data_item = self.data[str(index)]
        question = data_item['QUESTION']
        short_answs = [data_item["final_decision"]]
        long_answs = [data_item["LONG_ANSWER"]]

        short_label_list = []
        long_label_list = []
        for index, (short_answ, long_answ) in enumerate(zip(short_answs,long_answs)):
            short_label = self.LLM_tokenizer(short_answ, add_special_tokens=False)["input_ids"][:self.args.max_new_tokens]
            long_label = self.LLM_tokenizer(long_answ, add_special_tokens=False)["input_ids"][:self.args.max_new_tokens]

            short_label_list.append(short_label)
            long_label_list.append(long_label)

            if index==0:
                short_one_hot_label = torch.zeros(self.LLM_tokenizer.vocab_size).long()
                long_one_hot_label = torch.zeros(self.LLM_tokenizer.vocab_size).long()

                short_one_hot_label.index_fill_(0, torch.tensor(short_label), torch.tensor(1, dtype=torch.int64))
                long_one_hot_label.index_fill_(0, torch.tensor(long_label), torch.tensor(1, dtype=torch.int64))

        return {"question": question,  
                "answer": short_answs, "label": short_label_list, "one_hot_label": short_one_hot_label,
                "long_answer": long_answs, "long_label": long_label_list, "long_one_hot_label": long_one_hot_label,
                }



def collate_fn_PubmedQA(data):
    batch_data = {'question': [], "answer":[],  "label":[], "one_hot_label": [] , "long_answer":[],  "long_label":[], "long_one_hot_label": [] }
    for data_item in data:
        for k, v in batch_data.items():
            tmp = data_item[k]
            batch_data[k].append(tmp)
            
    batch_data['question'] = batch_data['question']

    batch_data['answer']   = batch_data['answer']
    batch_data['label']   = batch_data['label']
    batch_data['one_hot_label']   = torch.stack(batch_data['one_hot_label'])

    batch_data['long_answer']   = batch_data['long_answer']
    batch_data['long_label']   = batch_data['long_label']
    batch_data['long_one_hot_label']   = torch.stack(batch_data['long_one_hot_label'])

    return batch_data

 

def get_loader_PubmedQA(args, tokenizer, train_file_path, dev_file_path, test_file_path) :
    
    train_dataset = PubmedQA(args, train_file_path, tokenizer)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=args.train_batch_size,
                                   shuffle=False,
                                   pin_memory=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn_PubmedQA,
                                  )       

    # for demonstration
    dev_dataset = PubmedQA(args, dev_file_path, tokenizer)
    dev_data_loader = DataLoader(dataset=dev_dataset,
                                 shuffle=False,
                                 pin_memory=True,
                                 batch_size=args.test_batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn_PubmedQA,
                                ) 
    

    test_dataset = PubmedQA(args, test_file_path, tokenizer)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn_PubmedQA,
                                 )    

    return train_data_loader, dev_data_loader, test_data_loader, args



