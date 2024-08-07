from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from transformers import pipeline, GenerationConfig
from langchain.chains import LLMChain
import torch.nn.functional as F
import json
import os
import time
import csv
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from src.metrics import acc_PRF, F_EM
from torch.utils.tensorboard import SummaryWriter
from utils.utils import extracted_token_id_label, LineListOutputParser, empty_logger_file, combine_doc, __dist__, left_pad_loss_logit, get_logger


class My_Trainer:
    def __init__(self, args, MI_learner, my_gate, LLM, LLM_tokenizer, device, retriever):
        self.args = args
        self.print_logger = args.print_logger
        self.loss_fct = nn.NLLLoss(reduction="none")
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

        self.device = device
        self.MI_learner = MI_learner
        self.my_gate = my_gate

        self.LLM = LLM
        self.LLM_tokenizer = LLM_tokenizer

        # self.writer = SummaryWriter(args.dir_path+"/runs/")
        self.test_result_logger = get_logger(self.args.dir_path, "test_result")
        
        if self.args.RA_method in ["Only_RA", "Gate_RA", "Gate_MI_RA", "MI_RA"]:
            prompt_format = "retrieve-prompt"
            self.retriever =  retriever

            if self.args.RA_method == "Gate_RA":
                param_list_my_gate = list(self.my_gate.parameters())
                total_param = param_list_my_gate
            elif self.args.RA_method == "MI_RA" :
                param_list_MI_learner = list(self.MI_learner.parameters())
                total_param = param_list_MI_learner
            elif self.args.RA_method == "Gate_MI_RA":
                param_list_my_gate = list(self.my_gate.parameters())
                param_list_MI_learner = list(self.MI_learner.parameters())
                total_param = param_list_MI_learner+param_list_my_gate
            else:
                total_param = []

            if len(total_param) >0:
                self.optimizer = torch.optim.Adam(total_param, lr=self.args.lr, weight_decay=self.args.l2_coef)
                lr_lambda = lambda step: 1 if step < self.args.init_lr_num else self.args.lr_decay ** ((step - self.args.init_lr_num) // self.args.lr_decay_interval)
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            prompt_format = "general-prompt"

        self.print_logger.info(f"prompt_format: {prompt_format} \n")    
            
        with open(self.args.prompt_file, "r") as f:
            prompt_data = json.load(f)
            general_prompt_text = prompt_data["general-prompt"]
            prompt_text = prompt_data[prompt_format]

        self.prompt = PromptTemplate.from_template(prompt_text)
        self.general_prompt = PromptTemplate.from_template(general_prompt_text)

        if args.LLM == "chatGPT":
            self.pipe = LLMChain(prompt=self.prompt, llm=LLM)
            self.general_pipe = LLMChain(prompt=self.general_prompt, llm=LLM)

    def return_input_dict(self, data_item, retrieve_docs):
        if self.args.RA_method in ["Only_RA", "Gate_RA", "MI_RA", "Gate_MI_RA"]:
            if self.args.dataset in ["USMLE", "MedMCQA", "HEADQA"]:
                input_dict = {'question': data_item["question"], 'options': data_item["options"], "context": retrieve_docs}
            else:
                input_dict = {'question': data_item["question"],  "context": retrieve_docs}
        else:
            if self.args.dataset in ["USMLE", "MedMCQA", "HEADQA"]:
                input_dict = {'question': data_item["question"], 'options': data_item["options"]}
            else:
                input_dict = {'question': data_item["question"]}
        return input_dict
    
    def add_gold_retrieval(self, retrieve_docs, data_item):
        len_retrieve_docs = len(retrieve_docs)
        for index, (question_item, options_item, answer_item) in enumerate(zip(data_item['question'], data_item['options'], data_item['answer'])):
            for item in options_item.split('. '):
                if item.strip().startswith('<' + answer_item + '>'):
                    answer_str = item.split('>')[1].strip()
                    temp_doc = "document (0): " + question_item + " The answer is " + answer_str + "\n\n"
            
            if len_retrieve_docs>=1:
                retrieve_docs[index] =  temp_doc + retrieve_docs[index] 
            else:
                retrieve_docs.append(temp_doc) 

        return retrieve_docs
    
    def get_multi_query_input(self, questions, data_item):
        if self.args.multi_query:
            if "rewrite_question" in data_item.keys():
                if len(data_item["rewrite_question"])==0:
                    query = []
                    for question in questions:
                        query.append(question)
                        query.append(question)
                    return query
                else:
                    rewrite_questions = data_item["rewrite_question"]
                    query = []
                    for question, rewrite_question in zip(questions, rewrite_questions):
                        query.append(question)
                        query+=rewrite_question
                    return query
            else:
                query = []
                for question in questions:
                    query.append(question)
                    query.append(question)
                return query
        else:
            return questions

    def general_input_loop(self, data_item, labels, batch_answer):
        with torch.no_grad():
            if self.args.dataset in ["USMLE", "MedMCQA", "HEADQA"]:
                general_input_dict = {'question': data_item["question"], 'options': data_item["options"]}
            else:
                general_input_dict = {'question': data_item["question"]}
            
            general_batch_input_list, general_batch_pred, general_batch_id_pred, general_batch_hallucination_cnt, \
                general_save_doc_num, general_batch_loss_list, general_batch_logit_log_softmax, pior_list \
                = self.pipeline_inference(self.general_prompt, general_input_dict, labels, batch_answer, training_flag=True, record_flag=False)
        return general_batch_input_list, general_batch_pred, general_batch_id_pred, general_batch_hallucination_cnt, \
                general_save_doc_num, general_batch_loss_list, general_batch_logit_log_softmax, pior_list
    
    def RA_input_loop(self, input_dict, labels, batch_answer):
        with torch.no_grad():
            RA_batch_input_list, RA_batch_pred, RA_batch_id_pred, RA_batch_hallucination_cnt, RA_save_doc_num, RA_batch_loss, RA_batch_logit_log_softmax, pior_list \
                = self.pipeline_inference(self.prompt, input_dict, labels, batch_answer, training_flag=True, record_flag=False)
        
        return RA_batch_input_list, RA_batch_pred, RA_batch_id_pred, RA_batch_hallucination_cnt, \
                        RA_save_doc_num, RA_batch_loss, RA_batch_logit_log_softmax, pior_list

    def Only_RA(self, question, data_item, labels, batch_answer):
        query = self.get_multi_query_input(question, data_item)
        query_emb_list, attention_mask_list, bags_list = self.retriever.search_document(query) 
        retrieve_docs = combine_doc(bags_list)
        input_dict = self.return_input_dict(data_item, retrieve_docs)
        
        RA_batch_input_list, RA_batch_pred, RA_batch_id_pred, \
            RA_batch_hallucination_cnt, RA_save_doc_num, RA_batch_loss, RA_batch_logit_log_softmax, pior_list \
                = self.RA_input_loop(input_dict, labels, batch_answer)

        old_doc_len =  sum([len(i) for i in self.LLM_tokenizer(retrieve_docs)["input_ids"]]) / len(retrieve_docs)
        return  RA_batch_input_list, RA_batch_pred, RA_batch_id_pred, RA_batch_hallucination_cnt, \
            RA_batch_loss, RA_batch_logit_log_softmax, pior_list, old_doc_len
    
    def Gate_RA(self, total_gate_res, total_new_lable,
                   data_item, labels, batch_answer, question,
                   label_0_0=0, label_0_1=0, label_1_0=0, label_1_1=0):
        
        general_batch_input_list, general_batch_pred, general_batch_id_pred, \
            general_batch_hallucination_cnt, general_save_doc_num, general_batch_loss_list, \
                general_batch_logit_log_softmax, orginal_pior_list = self.general_input_loop(data_item, labels, batch_answer)

        query = self.get_multi_query_input(question, data_item)
        query_emb_list, attention_mask_list, bags_list = self.retriever.search_document(query) 
        retrieve_docs = combine_doc(bags_list)
        input_dict = self.return_input_dict(data_item, retrieve_docs)

        RA_batch_input_list, RA_batch_pred, RA_batch_id_pred, \
            RA_batch_hallucination_cnt, RA_save_doc_num, RA_batch_loss, RA_batch_logit_log_softmax, tra_pior_list \
                = self.RA_input_loop(input_dict, labels, batch_answer)
    

        raw_ques_emb_list = []
        raw_doc_emb_list = []
        for bag, raw_ques_emb, ques_att_mask in zip(bags_list, query_emb_list, attention_mask_list):
            with torch.no_grad():
                raw_doc_emb, _, doc_att_mask = self.retriever.embed_queries(self.args, bag)

            raw_ques_emb_list.append(raw_ques_emb[0])
            raw_doc_emb_list.append(torch.mean(raw_doc_emb[:, 0, :], dim=0))
        raw_ques_emb_list = torch.stack(raw_ques_emb_list)
        raw_doc_emb_list  = torch.stack(raw_doc_emb_list)

        gate_loss, gate_res, new_lable, new_label_count_list  = self.my_gate(general_batch_pred, RA_batch_pred, batch_answer, 
                                                                             RA_batch_loss, raw_ques_emb_list, 
                                                                             raw_doc_emb_list, general_batch_loss_list,  
                                                                             RA_batch_logit_log_softmax, general_batch_logit_log_softmax)

        label_0_0 += new_label_count_list[0]
        label_0_1 += new_label_count_list[1]
        label_1_0 += new_label_count_list[2]
        label_1_1 += new_label_count_list[3]

        total_gate_res+=gate_res.tolist()
        total_new_lable+=new_lable
        gate_acc = sum(1 for true_label, predicted_label in zip(total_new_lable, total_gate_res) if true_label == predicted_label) / len(total_new_lable)
        gate_acc = round(gate_acc, 2)

        return gate_loss, gate_acc, gate_res, \
            RA_batch_input_list,  RA_batch_pred, RA_batch_id_pred, RA_batch_hallucination_cnt, \
            general_batch_input_list, general_batch_hallucination_cnt, general_batch_id_pred, general_batch_pred,\
            label_0_0, label_0_1, label_1_0, label_1_1, total_gate_res, total_new_lable, orginal_pior_list,  tra_pior_list  

    def Train_MI_RA(self, question, data_item, labels, batch_answer, one_hot_labels):
        query = self.get_multi_query_input(question, data_item)
        query_emb_list, attention_mask_list, bags_list = self.retriever.search_document(query) 
        retrieve_docs = combine_doc(bags_list)
        input_dict = self.return_input_dict(data_item, retrieve_docs)
        
        RA_batch_input_list, RA_batch_pred, RA_batch_id_pred, \
            RA_batch_hallucination_cnt, RA_save_doc_num, RA_batch_loss, RA_batch_logit_log_softmax, pior_list \
                = self.RA_input_loop(input_dict, labels, batch_answer)

        loss_list, new_retrieve_docs, select_doc_num, raw_ques_emb_list, raw_doc_emb_list\
            = self.MI_learner(bags_list, query_emb_list, attention_mask_list, self.retriever, True, one_hot_labels, RA_batch_logit_log_softmax)
        
        old_doc_len =  sum([len(i) for i in self.LLM_tokenizer(retrieve_docs)["input_ids"]]) / len(retrieve_docs)
        new_doc_len =  sum([len(i) for i in self.LLM_tokenizer(new_retrieve_docs)["input_ids"]]) / len(new_retrieve_docs)      
        old_doc_len = round(old_doc_len, 2)
        new_doc_len = round(new_doc_len, 2)

        return loss_list, RA_batch_input_list, RA_batch_pred, RA_batch_id_pred, RA_batch_hallucination_cnt, \
            RA_batch_loss, RA_batch_logit_log_softmax, old_doc_len, new_doc_len, raw_ques_emb_list, raw_doc_emb_list, pior_list

    def Test_MI_RA(self, question, data_item, labels, batch_answer, one_hot_labels):
        query = self.get_multi_query_input(question, data_item)
        query_emb_list, attention_mask_list, bags_list = self.retriever.search_document(query) 
        retrieve_docs = combine_doc(bags_list)
        
        loss_list, new_retrieve_docs, select_doc_num, raw_ques_emb_list, raw_doc_emb_list\
            = self.MI_learner(bags_list, query_emb_list, attention_mask_list, self.retriever, False, "one_hot_labels", "RA_batch_logit_log_softmax")
        
        input_dict = self.return_input_dict(data_item, new_retrieve_docs)
        RA_batch_input_list, RA_batch_pred, RA_batch_id_pred, \
            RA_batch_hallucination_cnt, RA_save_doc_num, RA_batch_loss, RA_batch_logit_log_softmax, pior_list\
                = self.RA_input_loop(input_dict, labels, batch_answer)
        
        old_doc_len =  sum([len(i) for i in self.LLM_tokenizer(retrieve_docs)["input_ids"]]) / len(retrieve_docs)
        new_doc_len =  sum([len(i) for i in self.LLM_tokenizer(new_retrieve_docs)["input_ids"]]) / len(new_retrieve_docs)      
        old_doc_len = round(old_doc_len, 2)
        new_doc_len = round(new_doc_len, 2)

        return loss_list, RA_batch_input_list, RA_batch_pred, RA_batch_id_pred, RA_batch_hallucination_cnt, \
            RA_batch_loss, RA_batch_logit_log_softmax, old_doc_len, new_doc_len, raw_ques_emb_list, raw_doc_emb_list, pior_list
    
    def Gate_MI_RA(self, total_gate_res, total_new_lable,
                   data_item, labels, batch_answer, question, one_hot_labels,
                   label_0_0=0, label_0_1=0, label_1_0=0, label_1_1=0, train_flag=False):
        
        general_batch_input_list, general_batch_pred, general_batch_id_pred, \
            general_batch_hallucination_cnt, general_save_doc_num, general_batch_loss_list, \
                general_batch_logit_log_softmax, orginal_pior_list= self.general_input_loop(data_item, labels, batch_answer)

        if train_flag:
            loss_list, RA_batch_input_list, RA_batch_pred, RA_batch_id_pred, RA_batch_hallucination_cnt, \
                RA_batch_loss, RA_batch_logit_log_softmax, old_doc_len, new_doc_len, raw_ques_emb_list, raw_doc_emb_list, MI_pior_list\
                        = self.Train_MI_RA(question, data_item, labels, batch_answer, one_hot_labels)
        else:
            loss_list, RA_batch_input_list, RA_batch_pred, RA_batch_id_pred, RA_batch_hallucination_cnt, \
                RA_batch_loss, RA_batch_logit_log_softmax, old_doc_len, new_doc_len, raw_ques_emb_list, raw_doc_emb_list, MI_pior_list\
                        = self.Test_MI_RA(question, data_item, labels, batch_answer, one_hot_labels)
                
        gate_loss, gate_res, new_lable, new_label_count_list  = self.my_gate(general_batch_pred, RA_batch_pred, batch_answer, 
                                                                             RA_batch_loss, raw_ques_emb_list, 
                                                                            raw_doc_emb_list, general_batch_loss_list,  
                                                                            RA_batch_logit_log_softmax, general_batch_logit_log_softmax)

        label_0_0 += new_label_count_list[0]
        label_0_1 += new_label_count_list[1]
        label_1_0 += new_label_count_list[2]
        label_1_1 += new_label_count_list[3]

        total_gate_res+=gate_res.tolist()
        total_new_lable+=new_lable
        gate_acc = sum(1 for true_label, predicted_label in zip(total_new_lable, total_gate_res) if true_label == predicted_label) / len(total_new_lable)
        gate_acc = round(gate_acc, 2)

        return loss_list, gate_loss, gate_acc, old_doc_len, new_doc_len, gate_res, \
            RA_batch_input_list,  RA_batch_pred, RA_batch_id_pred, RA_batch_hallucination_cnt, \
            general_batch_input_list, general_batch_hallucination_cnt, general_batch_id_pred, general_batch_pred,\
            label_0_0, label_0_1, label_1_0, label_1_1, total_gate_res, total_new_lable, orginal_pior_list, MI_pior_list

    def train_proc(self, train_data_loader, test_data_loader):
        self.print_logger.info("Start training ... \n ")
        
        total_batch = len(train_data_loader)
        step_num = -1
        best_step = 0
        best_step_in = 0
        best_performce = 0
        best_performce_in = 0
        eval_num = 0
        for epoch_num in range(99999999):
            label_0_0, label_0_1, label_1_0, label_1_1 = 0, 0, 0, 0

            total_gate_res = []
            total_new_lable = []
            # self.train_result_logger = get_logger(self.args.dir_path, "train_result")
            
            for tmp_step, data_item in enumerate(train_data_loader):
                step_num+=1
                question = data_item['question']   
                labels = data_item['label']
                one_hot_labels = data_item['one_hot_label']
                batch_answer = data_item["answer"]
              
                if self.args.RA_method == "Gate_RA":
                    gate_loss, gate_acc, _, _,  _, _, _, _, _, _, _, \
                        label_0_0, label_0_1, label_1_0, label_1_1, \
                              total_gate_res, total_new_lable, orginal_pior_list,  tra_pior_list \
                        = self.Gate_RA(total_gate_res, total_new_lable, data_item, labels, 
                                       batch_answer, question, label_0_0, label_0_1, label_1_0, label_1_1)
                    gate_loss.backward()
                    
                    if tmp_step% 50==0:
                        self.print_logger.info(f"epoch_num: {epoch_num}, training process num: {step_num}/{total_batch},  gate_loss: {round(float(gate_loss), 4)},  \
                                            \n gate_acc:{gate_acc}, label_0_0:{label_0_0}, label_0_1:{label_0_1}, label_1_0:{label_1_0}, label_1_1:{label_1_1}, \
                                            \n\n best_step:{best_step}, best_performce: {best_performce}, \
                                            \n best_step_in:{best_step_in}, best_performce_in: {best_performce_in} \n ")
                    
                elif self.args.RA_method == "MI_RA":
                    loss_list, _, _, _, _, _, _, old_doc_len, new_doc_len, _, _, pior_list \
                                = self.Train_MI_RA(question, data_item, labels, batch_answer, one_hot_labels)
                    
                    loss_list[-1].backward()
                    if tmp_step%50==0:
                        self.print_logger.info(f"epoch_num: {epoch_num}, training process num: {step_num}/{total_batch},  kl_soft_loss: {round(float(loss_list[1]), 4)}, kl_hard_loss: {round(float(loss_list[2]), 4)}, len_loss: {round(float(loss_list[0]), 4)}, \
                                            \n old_doc_len:{old_doc_len}, new_doc_len:{new_doc_len}, \
                                            \n\n best_step:{best_step}, best_performce: {best_performce}, \
                                            \n best_step_in:{best_step_in}, best_performce_in: {best_performce_in} \n ")
                    
                elif self.args.RA_method == "Gate_MI_RA":
                    loss_list, gate_loss, gate_acc, old_doc_len, new_doc_len, _, _, _, _, _, _, _, _, _, \
                        label_0_0, label_0_1, label_1_0, label_1_1, _, _, orginal_pior_list, MI_pior_list = self.Gate_MI_RA(total_gate_res, total_new_lable,
                                                                    data_item, labels, batch_answer, question, one_hot_labels,
                                                                    label_0_0, label_0_1, label_1_0, label_1_1, True)
                    loss_list[-1].backward()
                    gate_loss.backward()

                    if tmp_step%50==0:
                        self.print_logger.info(f"epoch_num: {epoch_num}, training process num: {step_num}/{total_batch},  gate_loss: {round(float(gate_loss), 4)}, kl_soft_loss: {round(float(loss_list[1]), 4)}, kl_hard_loss: {round(float(loss_list[2]), 4)}, len_loss: {round(float(loss_list[0]), 4)}, \
                                        \n gate_acc:{gate_acc}, old_doc_len:{old_doc_len}, new_doc_len:{new_doc_len}, label_0_0:{label_0_0}, label_0_1:{label_0_1}, label_1_0:{label_1_0}, label_1_1:{label_1_1}, \
                                        \n\n best_step:{best_step}, best_performce: {best_performce}, \
                                        \n best_step_in:{best_step_in}, best_performce_in: {best_performce_in} \n ")
        
                else:
                    raise Exception("no need train !")

                    
                if (step_num + 1) % self.args.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (step_num % self.args.train_eval==0) and step_num>1 and step_num < self.args.total_step:
                    eval_num +=1
                    # self.train_result_logger = empty_logger_file(self.train_result_logger)

                    break_cnt = 2 if self.args.test_code_flag else None
                    with torch.no_grad():
                        test_performce, test_performce_in, all_test_predictions, all_test_input_list, \
                            all_test_answers, orginal_pior_list, r3_pior_list, old_doc_len,  new_doc_len = self.test_proc(test_data_loader, step_num, break_cnt=break_cnt)

                    if test_performce>=best_performce:
                        best_performce = test_performce
                        best_step = step_num

                        tmp_result = [1 if a == b else 0 for a, b in zip(all_test_answers, all_test_predictions)]
                        with open(self.args.dir_path+'/MI_' +str(best_performce)+'.txt', "w") as f:
                            f.writelines("orginal_pior_list\n")
                            f.writelines(str(orginal_pior_list)+"\n")

                            f.writelines("r3_pior_list\n")
                            f.writelines(str(r3_pior_list)+"\n")

                            f.writelines("right_wrong\n")
                            f.writelines(str(tmp_result)+"\n")

                    if test_performce_in>=best_performce_in:
                        best_performce_in = test_performce_in
                        best_step_in = step_num

                        self.test_result_logger = get_logger(self.args.dir_path, "test_result")
                        for batch_pred, batch_input, batch_answer in zip(all_test_predictions, all_test_input_list, all_test_answers):
                            self.recored_res(batch_pred, batch_input, batch_answer, training_flag=False, record_flag=True)     

                        tmp_result = [1 if a == b else 0 for a, b in zip(all_test_answers, all_test_predictions)]
                        with open(self.args.dir_path+'/in_MI_' +str(best_performce_in)+'.txt', "w") as f:
                            f.writelines("orginal_pior_list\n")
                            f.writelines(str(orginal_pior_list)+"\n")

                            f.writelines("r3_pior_list\n")
                            f.writelines(str(r3_pior_list)+"\n")

                            f.writelines("right_wrong\n")
                            f.writelines(str(tmp_result)+"\n")

                if step_num == self.args.total_step :
                    with open(self.args.dir_path+'/Finsh' +'.txt', "w") as f:
                            f.writelines(" ")

                break_cnt = 2 if self.args.test_code_flag else None
                if break_cnt is not None and break_cnt<step_num:
                    break
                        
    def test_proc(self, test_data_loader, eval_num=0, break_cnt=None):
            
        self.print_logger.info("\n Start test ...  ")
        
        start_time = time.time()

        all_test_labels = []
        all_test_prediction_ids = []
        all_test_answers = []
        all_test_input_list = []
        all_test_predictions = []
        all_hallucination = []
        total_old_doc_len, total_new_doc_len =0, 0

        all_orginal_pior_list = []
        all_r3_pior_list = []

        total_new_lable =[]
        total_gate_res = []
        for index, data_item in enumerate(test_data_loader):
            if index%200==0:
                self.print_logger.info(f"testing process num: {index}")
            question = data_item['question']
            batch_answer = data_item["answer"]
            labels = data_item['label']
            one_hot_labels = data_item['one_hot_label']

            all_test_answers += batch_answer
            all_test_labels+=labels
            if self.args.RA_method == "Gate_RA":
                # if self.args.infer_add_gold_retrieval:
                #     retrieve_docs = self.add_gold_retrieval(retrieve_docs, data_item)

                gate_loss, gate_acc, gate_res, \
                RA_batch_input_list,  RA_batch_pred, RA_batch_id_pred, RA_batch_hallucination_cnt, \
                general_batch_input_list, general_batch_hallucination_cnt, general_batch_id_pred, general_batch_pred,\
                label_0_0, label_0_1, label_1_0, label_1_1, total_gate_res, total_new_lable, orginal_pior_list,  gate_pior_list   \
                    = self.Gate_RA(total_gate_res, total_new_lable, data_item, labels, batch_answer, question)
                
                all_orginal_pior_list+=orginal_pior_list
                
                for gate_index, res in enumerate(gate_res):
                    if res ==1:
                        all_test_input_list+=RA_batch_input_list
                        all_hallucination.append(RA_batch_hallucination_cnt[gate_index])
                        all_test_prediction_ids.append(RA_batch_id_pred[gate_index])
                        all_test_predictions.append(RA_batch_pred[gate_index])
                        try:
                            all_r3_pior_list.append(gate_pior_list[gate_index])
                        except:
                            print(1)
                    else:
                        retrieve_docs = ""
                        all_test_input_list+=general_batch_input_list
                        all_hallucination.append(general_batch_hallucination_cnt[gate_index])
                        all_test_prediction_ids.append(general_batch_id_pred[gate_index])
                        all_test_predictions.append(general_batch_pred[gate_index])
                        all_r3_pior_list.append(orginal_pior_list[gate_index])
                
            elif self.args.RA_method == "MI_RA":
                # if self.args.infer_add_gold_retrieval:
                #     retrieve_docs = self.add_gold_retrieval(retrieve_docs, data_item)
                
                loss_list, batch_input_list, batch_pred, batch_id_pred, batch_hallucination_cnt, \
                    batch_loss, batch_logit_log_softmax, old_doc_len, new_doc_len, raw_ques_emb_list, raw_doc_emb_list, pior_list \
                                = self.Test_MI_RA(question, data_item, labels, batch_answer, one_hot_labels)
                
                total_old_doc_len += old_doc_len
                total_new_doc_len += new_doc_len

                all_test_input_list+=batch_input_list
                all_hallucination += batch_hallucination_cnt
                all_test_prediction_ids+=batch_id_pred
                all_test_predictions+=batch_pred

            elif self.args.RA_method == "Gate_MI_RA":
                # if self.args.infer_add_gold_retrieval:
                #     retrieve_docs = self.add_gold_retrieval(retrieve_docs, data_item)

                MI_learner_loss, gate_loss, gate_acc, old_doc_len, new_doc_len, gate_res, \
                    RA_batch_input_list, RA_batch_pred, RA_batch_id_pred, RA_batch_hallucination_cnt, \
                        general_batch_input_list, general_batch_hallucination_cnt, general_batch_id_pred, general_batch_pred, \
                        _, _, _, _, total_gate_res, total_new_lable, orginal_pior_list, MI_pior_list \
                        = self.Gate_MI_RA(total_gate_res, total_new_lable, data_item, labels, batch_answer, question, one_hot_labels)

                all_orginal_pior_list+=orginal_pior_list
             

                total_old_doc_len += old_doc_len
                total_new_doc_len += new_doc_len
                
                for gate_index, res in enumerate(gate_res):
                    if res ==1:
                        all_test_input_list+=RA_batch_input_list
                        all_hallucination.append(RA_batch_hallucination_cnt[gate_index])
                        all_test_prediction_ids.append(RA_batch_id_pred[gate_index])
                        all_test_predictions.append(RA_batch_pred[gate_index])
                        all_r3_pior_list.append(MI_pior_list[gate_index])
                    else:
                        retrieve_docs = ""
                        all_test_input_list+=general_batch_input_list
                        all_hallucination.append(general_batch_hallucination_cnt[gate_index])
                        all_test_prediction_ids.append(general_batch_id_pred[gate_index])
                        all_test_predictions.append(general_batch_pred[gate_index])
                        all_r3_pior_list.append(orginal_pior_list[gate_index])

            elif self.args.RA_method == "Only_RA":
                batch_input_list, batch_pred, batch_id_pred, batch_hallucination_cnt, \
                    batch_loss, batch_logit_log_softmax, pior_list, old_doc_len = self.Only_RA(question, data_item, labels, batch_answer)
                
                total_old_doc_len += old_doc_len

                all_r3_pior_list+=pior_list

                all_test_input_list+=batch_input_list
                all_hallucination+=batch_hallucination_cnt
                all_test_prediction_ids+=batch_id_pred
                all_test_predictions+=batch_pred

            elif self.args.RA_method == "No_RA":
                batch_input_list, batch_pred, batch_id_pred, \
                    batch_hallucination_cnt, general_save_doc_num, general_batch_loss_list, \
                        general_batch_logit_log_softmax, pior_list = self.general_input_loop(data_item, labels, batch_answer)
                
                all_orginal_pior_list+=pior_list
                
                all_test_input_list+=batch_input_list
                all_hallucination+=batch_hallucination_cnt
                all_test_prediction_ids+=batch_id_pred
                all_test_predictions+=batch_pred
            
            else:
                raise Exception("no need train !")

            break_cnt = 2 if self.args.test_code_flag else None
            if break_cnt is not None and break_cnt<index:
                break
        
        old_doc_len = round(total_old_doc_len / len(test_data_loader), 2)
        new_doc_len = round(total_new_doc_len / len(test_data_loader), 2)  

        if self.args.dataset in ["USMLE", "MedMCQA", "HEADQA"]:
            test_acc, test_precision, test_recall, test_f1 = acc_PRF(all_test_labels, all_test_prediction_ids)
            all_hallucination = round(sum(all_hallucination)/len(test_data_loader.dataset)*100, 2)
            self.args.print_logger.info(f"test: acc {test_acc}, f1 {test_f1}, precision {test_precision}, recall {test_recall}, old_doc_len:{old_doc_len}, new_doc_len:{new_doc_len}, hallucination: {all_hallucination} ")
            record_performance = test_acc
            record_performance_in = test_acc
        else:
            test_f1, test_EM, test_EM_in = F_EM(all_test_answers, all_test_predictions)
            self.args.print_logger.info(f"test: f1 {test_f1}, EM : {test_EM}/{test_EM_in}, old_doc_len:{old_doc_len}, new_doc_len:{new_doc_len}")
            record_performance = test_EM
            record_performance_in =  test_EM_in

        cost_time  = (time.time() - start_time)/60
        if self.args.RA_method in ["Gate_RA", "Gate_MI_RA"]:
            self.args.print_logger.info(f"cost_time: {cost_time} , gate_res_list: { round(sum(total_gate_res) / len(total_gate_res), 2) }, {sum(total_gate_res)} / {len(total_gate_res)} \n ")
        else:
            self.args.print_logger.info(f"cost_time: {cost_time}  \n ")

        # self.writer.add_scalar('Performance/test/acc', test_acc, eval_num )
        # self.writer.add_scalar('Performance/test/precision', test_precision, eval_num )
        # self.writer.add_scalar('Performance/test/recall', test_recall, eval_num )
        # self.writer.add_scalar('Performance/test/f1', test_f1, eval_num )

        return record_performance, record_performance_in, all_test_predictions, \
            all_test_input_list, all_test_answers, all_orginal_pior_list, all_r3_pior_list, old_doc_len,  new_doc_len

         
    def pipeline_inference(self, used_prompt, input_dict, label, batch_answer, training_flag=False, record_flag=True):
        if self.args.LLM == "chatGPT":
            batch_pred, batch_id_pred, batch_hallucination_cnt, save_doc_num = self.non_local_llm_infer(input_dict, label, batch_answer, training_flag, record_flag)
            batch_loss, batch_logit_log_softmax = 0, 0
        else:
            batch_my_input_list, batch_pred, batch_id_pred, batch_hallucination_cnt, save_doc_num, batch_loss_list, batch_logit_log_softmax, pior_list= self.local_llm_infer(used_prompt, input_dict, label, batch_answer, training_flag, record_flag)

        return batch_my_input_list, batch_pred, batch_id_pred, batch_hallucination_cnt, save_doc_num, batch_loss_list, batch_logit_log_softmax, pior_list

    def non_local_llm_infer(self, input_dict, label, batch_answer, training_flag=False, record_flag=True):
        batch_pred = []
        batch_id_pred = []
        keys = input_dict.keys()
        batch_hallucination_cnt = 0
        
        save_doc_num = [self.args.n_docs]*len(label)

        for index2, values in enumerate(zip(*input_dict.values())):
            current_inputs = dict(zip(keys, values))
            try:
                pred = self.pipe(current_inputs)
            except:
                current_inputs["context"] = "document ".join(current_inputs["context"].split("document")[:-1])
                self.print_logger.info("too long context, we short one retrieval results !")
                save_doc_num[index2] = save_doc_num[index2]-1
                try:
                    pred = self.pipe(current_inputs)
                except:
                    current_inputs["context"] = "document ".join(current_inputs["context"].split("document")[:-1])
                    self.print_logger.info("too long context, we short two retrieval results !")
                    save_doc_num[index2] = save_doc_num[index2]-1
                    try:
                        pred = self.pipe(current_inputs)
                    except:
                        current_inputs["context"] = "document ".join(current_inputs["context"].split("document")[:5])
                        self.print_logger.info("too long context for many times, we only take first 5 retrieval results !")
                        pred =  self.pipe(current_inputs)
                        save_doc_num[index2] = 5

            pred = pred["text"]
            my_input = self.prompt.format(**current_inputs)
            pred, id_pred, hallucination_cnt = extracted_token_id_label(my_input , label[index2], pred, batch_answer[index2], training_flag, record_flag) 

            batch_pred.append(pred)  
            batch_id_pred.append(id_pred)
            batch_hallucination_cnt+=hallucination_cnt

        return batch_pred, batch_id_pred, batch_hallucination_cnt, save_doc_num
    
    def local_llm_infer(self, used_prompt, input_dict, labels, batch_answer, training_flag=False, record_flag=True):
        save_doc_num = [self.args.n_docs]*len(labels)
       
        my_input_list = []
        keys = input_dict.keys()
        for values in zip(*input_dict.values()):
            current_inputs = dict(zip(keys, values))
            my_input = used_prompt.format(**current_inputs)
            my_input_list.append(my_input)
        
        batch_id_pred = []
        batch_pred = []
        batch_hallucination_cnt = []

        inputs = self.LLM_tokenizer(my_input_list, return_tensors="pt", padding=True).to(self.args.device)
        generation_config = GenerationConfig(
            max_new_tokens=self.args.max_new_tokens,  
            pad_token_id = self.LLM_tokenizer.eos_token_id,
            do_sample=self.args.do_sample,
            num_return_sequences=1, 
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            length_penalty=self.args.length_penalty,
            # num_beams=self.args.num_beams,
            return_dict_in_generate=True, 
            output_scores=True,
        )

        outputs = self.LLM.generate(**inputs, generation_config=generation_config)
        outputs_scores = torch.stack(outputs["scores"]).permute(1,0,2)
        batch_loss_list = []
        logit_log_softmax = []

        pior_list = []
        for index, (outputs_score, output, answer, label) in enumerate(zip(outputs_scores, outputs["sequences"], batch_answer, labels)):
            generation = self.LLM_tokenizer.decode(output, skip_special_tokens=True)
            pred, id_pred, hallucination_cnt = extracted_token_id_label(generation, label, self.LLM_tokenizer, self.args.dataset, used_prompt, self.args.LLM)

            if self.args.dataset in ["USMLE", "MedMCQA", "HEADQA"]:
                label = torch.LongTensor([label]).to(outputs_score[0].device)
                process_score =  F.log_softmax(outputs_score[:len(label)], dim=-1)
                batch_loss_list.append( self.loss_fct(process_score, label.view(-1)) )

                pior_list.append( round(float(F.softmax(outputs_score[0], dim=0).detach()[int(label)]), 2) )
            else:
                temp_loss = []
                temp_process_score = []
                
                pior_list.append( round(float(F.softmax(outputs_score[0], dim=0).detach()[int(label[0][0])]), 2) )
                for lab in label:
                    lab = torch.LongTensor([lab[:self.args.max_new_tokens]]).to(outputs_score[0].device)

                    need_len = min(len(lab[0]), outputs_score.size(0))
                    process_score =  F.log_softmax(outputs_score[:need_len], dim=-1)
                    temp_loss.append(torch.mean(self.loss_fct(process_score, lab.view(-1)[:need_len])))
                    
                    temp_process_score.append(torch.sum(process_score, dim=0))

                batch_loss_list.append( torch.mean(torch.stack(temp_loss)).reshape(1) )
                process_score = torch.mean(torch.stack(temp_process_score), dim=0).unsqueeze(0)

            logit_log_softmax.append(process_score)
            batch_pred.append(pred)
            batch_id_pred.append(id_pred)
            batch_hallucination_cnt.append(hallucination_cnt)

        logit_log_softmax = torch.stack(logit_log_softmax)
        batch_loss_list = torch.stack(batch_loss_list)
        return my_input_list, batch_pred, batch_id_pred, batch_hallucination_cnt, save_doc_num, batch_loss_list, logit_log_softmax, pior_list

    def recored_res(self, pred, my_input, answer, training_flag, record_flag):
        if training_flag:
            # result_logger = self.train_result_logger
            pass
        else:    
            result_logger = self.test_result_logger

        if record_flag:

            result_logger.info(f"my_input: {my_input}")
            result_logger.info(f"answer:   {answer} ")
            result_logger.info(f"pred:   {pred} ")
            result_logger.info(f"=================================================================================================================================================================================================\n\n")
            # result_logger.info(f"label:   {[self.LLM_tokenizer._convert_id_to_token(int(label_i))   for label_i in label] } ")
            # result_logger.info(f"id_pred: {[self.LLM_tokenizer._convert_id_to_token(id_pred_i) for id_pred_i in id_pred] } "+ "\n========================================================================================================================")
        
   