import logging
from datetime import datetime
import sys
import math
import torch
import os
import re
import time
import random
import shutil
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import List
from langchain.chat_models import ChatOpenAI
import torch.nn.functional as F


def left_pad_loss_logit(label, logit_log_softmax, max_len, pad_id ):

    padded_labels = torch.full((len(label), max_len), pad_id)
    for i, seq in enumerate(label):
        # padded_labels[i, :len(seq)] = torch.tensor(seq, dtype=torch.long).to(logit_log_softmax.device)
        padded_labels[i, -len(seq):] = torch.tensor(seq, dtype=torch.long).to(logit_log_softmax.device)

    need_len = max_len - logit_log_softmax.size(1)
    pad_tensor = torch.full((logit_log_softmax.size(0), need_len, logit_log_softmax.size(-1)), pad_id).to(logit_log_softmax.device)
    logit_log_softmax = torch.cat((pad_tensor, logit_log_softmax, ), dim=1)
    
    label_mask = (padded_labels != pad_id).to(logit_log_softmax.device)
    return padded_labels.to(logit_log_softmax.device), logit_log_softmax, label_mask

def map_one_hot_labels(labels, LLM_tokenizer):
    dic_map = {0:"A", 1:"B", 2:"C", 3:"D"}
    bz = len(labels)
    one_hot_label = torch.zeros(bz, LLM_tokenizer.vocab_size)
    ids= [LLM_tokenizer._convert_token_to_id(dic_map[label]) for label in labels]
    
    for index, id in enumerate(ids):
        one_hot_label[index][id] = 1
    return one_hot_label

def __dist__(x, y, dim=-1, tau=1, method='dot'): 
    if method == 'dot':
        sim = torch.matmul(x, y) / tau
    elif method == 'euclidean':
        x= x.unsqueeze(1)
        y= y.permute(1,0).unsqueeze(0)
        sim = (torch.pow(x - y, 2)).sum(dim) / tau
    elif method == 'cosine':
        x= x.unsqueeze(1)
        y= y.permute(1,0).unsqueeze(0)
        sim = torch.abs(F.cosine_similarity(x, y, dim=dim) / tau)
    elif method == 'KL':
        kl_mean_1 = F.kl_div(F.log_softmax(x, dim=-1), F.softmax(y, dim=-1), reduction='sum')
        kl_mean_2 = F.kl_div(F.log_softmax(y, dim=-1), F.softmax(x, dim=-1), reduction='sum')
        sim = (kl_mean_1 + kl_mean_2)/2
    return sim

class IndexRefreshScheduler(object):
    def __init__(self, format_str: str, freeze_retriever_steps: int, train_retriever: bool, logger):
        """Build an index refresh scheduler

        format_str: string that specifies the schedule.
            has the format: startstep-endstep:refreshrate,startstep-endstep:refreshrate
            e.g. format_str="0-100:10,100-1000000:500" will refresh the index every 10 steps for the first 100 steps
            and then every 500 steps from step 100 to 1M.

            Syntactic Sugar for a fixed schedule: can just pass in a single number
            e.g. format_str="100" will refresh the index every 100 steps

            -1 to never refresh
        )
        """
        self.logger = logger
        self.format_str = format_str
        self.train_retriever = train_retriever
        self.freeze_retriever_steps = freeze_retriever_steps
        self.steps2rates = IndexRefreshScheduler.parse_index_refresh_schedule_string(format_str)

    @classmethod
    def parse_index_refresh_schedule_string(cls, format_str):
        parsed = []
        if format_str == "-1":
            parsed = [(0, 2**32, 2**32)]
        elif format_str.isdigit():
            parsed = [(0, 2**32, int(format_str))]
        else:
            for piece in format_str.split(","):
                startend, rate = piece.split(":")
                start, end = startend.split("-")
                parsed.append((int(start), int(end), int(rate)))
        return parsed

    def is_time_to_refresh(self, step):
        if not (self.train_retriever or step == 0):  # if retriever is not trained only refresh at step 0
            return False
        if not step == 0 and step < self.freeze_retriever_steps:  # freeze first steps
            return False
        for st, en, rate in self.steps2rates:
            if st <= step < en:
                steps_since_refresh_schedule_change = step - st
                return (steps_since_refresh_schedule_change % rate) == 0
        self.logger.warn(
            "cant calculate refresh rate for this step, I dont have data here"
            " its likely training step is higher than the specificed refresh rate see --index_refresh_rate for help."
        )
        return False

class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")

class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return lines

def extracted_token_id_label(res, label, tokenizer, dataset, prompt, LLM):
    if dataset in ["USMLE", "MedMCQA", "HEADQA"]:
        res = res[-3:]
        label_list = [tokenizer._convert_token_to_id_with_added_voc("A"), tokenizer._convert_token_to_id_with_added_voc("B"),tokenizer._convert_token_to_id_with_added_voc("C"),tokenizer._convert_token_to_id_with_added_voc("D")]
        
        if "A" in res:
            return "A", [label_list[0]], 0 
        if "B" in res:
            return "B", [label_list[1]], 0
        if "C" in res:
            return "C", [label_list[2]], 0
        if "D" in res:
            return "D", [label_list[3]], 0
        
        if int(label[0]) in label_list:
            label_list.remove(int(label[0]))

        hall_label = random.choice(label_list)
        return res, [hall_label] , 1 

    else:
        
        res = res.split(prompt.template[-10:])[1].strip()
        label = tokenizer(res, add_special_tokens=False)["input_ids"]

        return res, label , 0

   
    

  
    
def remove_substring_and_after(original_string, substring):
    index = original_string.find(substring)
    if index != -1:
        return original_string[:index]
    else:
        return original_string 

def combine_doc(batch_retrieve_doc):
    tmp_str_List = []
    for retrieve_doc in batch_retrieve_doc:
        tmp_str = ""
        if len(retrieve_doc)>0:
            for index, i in enumerate(retrieve_doc):
                tmp_str += "document ("+ str(index+1) + ") \n\n"
                tmp_str = tmp_str + i + "\n\n"
        tmp_str_List.append(tmp_str)
    return tmp_str_List

def map_prob(labels, scores, LLM_tokenizer):    
    total_tmp_list = []
    dic_map = {0:"A", 1:"B", 2:"C", 3:"D"}
    prob_list = []
    for label, score in zip(labels, scores):
        id = LLM_tokenizer._convert_token_to_id(dic_map[label])
        prob = score.squeeze()[id]
        if math.isinf(prob):
            prob = torch.tensor(1e-9)
        prob_list.append(prob)
    return torch.stack(prob_list)  

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def make_log_dir():
    sys_path = str(sys.argv[1:])[1:-1].replace("'", "").replace("--", "").replace(",", "_").replace(" ", "") if len(str(sys.argv[1:]))>2 else str(sys.argv[1:])
    dir_path ="./results/output/"+sys_path[:80].replace("/","_")

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)

    return dir_path

def empty_logger_file(logger):
    file_handler = logger.handlers[0]
    log_file = logger.handlers[0].baseFilename

    logger.removeHandler(file_handler)
    file_handler.close()

    # 然后，清空文件内容
    with open(log_file, 'w') as file:
        pass  # 打开文件后立即关闭，内容被清空

    # 最后，重新创建FileHandler并绑定到logger
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)
    return logger

def get_logger(dir, name):
    
    logger = logging.getLogger(name)
    logger.handlers.clear()
    
    if name == "test_result":
        # 创建一个handler，用于写入日志文件
        filename = f'{datetime.now().date()}_{name}.log'
        filename = os.path.join(dir, filename)
        fh_test = logging.FileHandler(filename, mode='w+', encoding='utf-8')

    if name == "train_result":
        # 创建一个handler，用于写入日志文件
        filename = f'{datetime.now().date()}_{name}.log'
        filename = os.path.join(dir, filename)
        fh_train = logging.FileHandler(filename, mode='w+', encoding='utf-8')

    if name =="print":
        # 创建一个handler，用于写入日志文件
        filename = f'{datetime.now().date()}_{name}.log'
        filename = os.path.join(dir, filename)
        fh = logging.FileHandler(filename, mode='w+', encoding='utf-8')

        # 再创建一个handler用于输出到控制台
        ch = logging.StreamHandler()

    # 定义输出格式(可以定义多个输出格式例formatter1，formatter2)
    # formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    formatter = logging.Formatter('%(asctime)s %(message)s')

    # 定义日志输出层级
    logger.setLevel(logging.DEBUG)
    
    if name =="test_result":
        # 定义控制台输出层级
        # logger.setLevel(logging.DEBUG)
        # 为文件操作符绑定格式（可以绑定多种格式例fh.setFormatter(formatter2)）
        fh_test.setFormatter(formatter)
        # 给logger对象绑定文件操作符
        logger.addHandler(fh_test)

    if name =="train_result":
        # 定义控制台输出层级
        # logger.setLevel(logging.DEBUG)
        # 为文件操作符绑定格式（可以绑定多种格式例fh.setFormatter(formatter2)）
        fh_train.setFormatter(formatter)
        # 给logger对象绑定文件操作符
        logger.addHandler(fh_train)

    if name =="print":
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch.setFormatter(formatter)

        # 给logger对象绑定文件操作符
        logger.addHandler(ch)

    return logger

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory

def load_LLM(args, dtype=torch.bfloat16):
    if args.LLM == "chatGPT":
        model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=args.temperature) 
        tokenizer = AutoTokenizer.from_pretrained("../LLM_models/llama2/Llama-2-7b-chat-hf", use_fast=False)
        return model, tokenizer
    else:
        model_name_or_path = os.path.join(args.LLM)
        if model_name_or_path == "meta-llama/Llama-2-7b-chat-hf":
            model_name_or_path = "../LLM_models/llama2/Llama-2-7b-chat-hf"

        # Load the FP16 modelargs.triever
        args.print_logger.info(f"Loading {model_name_or_path} in {dtype}...")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map='auto',
            torch_dtype=dtype,
            # max_memory=get_max_memory(),
            load_in_8bit=False,
            offload_folder=model_name_or_path,
        )

        args.print_logger.info("Finish loading in %.2f mins." % ((time.time() - start_time)/60))

        # Load the tokenizer
        # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.cls_token = tokenizer.bos_token
        tokenizer.sep_token = tokenizer.eos_token
        
        return model, tokenizer
