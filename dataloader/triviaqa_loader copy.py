import json
import numpy as np
import pandas as pd
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import os


class TriviaQA(Dataset):

    def __init__(self, args, data_file, tokenizer):
        self.args = args
        self.LLM_tokenizer = tokenizer
        
        self.data = []
        with open(data_file, 'r') as file:
            for i in file:
                self.data.append(json.loads(i))

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        data_item = self.data[index]
        question = data_item['question']
        answs = data_item['answer']

        label_list = []

        for index, answ in enumerate(answs):
            label = self.LLM_tokenizer(answ, add_special_tokens=False)["input_ids"][:self.args.max_new_tokens]
            label_list.append(label)
            if index==0:
                one_hot_label = torch.zeros(self.LLM_tokenizer.vocab_size)
                one_hot_label.index_fill_(0, torch.tensor(label), torch.tensor(1))

        return {"question": question,  "answer": answs, "label": label_list, "one_hot_label": one_hot_label}



def collate_fn_TriviaQA(data):
    batch_data = {'question': [], "answer":[],  "label":[], "one_hot_label": [] }
    for data_item in data:
        for k, v in batch_data.items():
            tmp = data_item[k]
            batch_data[k].append(tmp)
            
    batch_data['question'] = batch_data['question']
    batch_data['answer']   = batch_data['answer']
    batch_data['label']   = batch_data['label']
    batch_data['one_hot_label']   = torch.stack(batch_data['one_hot_label'])
    return batch_data

 

def get_loader_TriviaQA(args, tokenizer, train_file_path, dev_file_path, test_file_path) :
    
    train_dataset = TriviaQA(args, train_file_path, tokenizer)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=args.train_batch_size,
                                   shuffle=False,
                                   pin_memory=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn_TriviaQA,
                                  )       

    # for demonstration
    dev_dataset = TriviaQA(args, dev_file_path, tokenizer)
    dev_data_loader = DataLoader(dataset=dev_dataset,
                                 shuffle=False,
                                 pin_memory=True,
                                 batch_size=args.test_batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn_TriviaQA,
                                ) 
    

    test_dataset = TriviaQA(args, test_file_path, tokenizer)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn_TriviaQA,
                                 )    

    return train_data_loader, dev_data_loader, test_data_loader, args



