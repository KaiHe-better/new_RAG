from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from transformers import pipeline
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
from src.metrics import My_Metrics
from torch.utils.tensorboard import SummaryWriter
from utils.utils import extracted_token_id_label, LineListOutputParser, empty_logger_file, combine_doc, __dist__, left_pad_loss_logit


class My_Trainer:

    def __init__(self, args, MI_learner, LLM, LLM_tokenizer, device, retriever):
        self.args = args
        self.print_logger = args.print_logger
        self.test_result_logger = args.test_result_logger
        self.train_result_logger = args.train_result_logger

        self.device = device
        self.MI_learner = MI_learner

        self.LLM = LLM
        self.LLM_tokenizer = LLM_tokenizer
        self.LLM_tokenizer.pad_token_id = self.LLM_tokenizer.eos_token_id
        self.LLM.config.pad_token_id = self.LLM_tokenizer.eos_token_id
        self.my_metrics = My_Metrics()
        self.writer = SummaryWriter(args.dir_path+"/runs/")

        if self.args.if_train:
            self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

        if self.args.if_RA or args.if_MI_RA:
            self.retriever =  retriever

            if self.args.if_train:
                param_list_MI_learner = list(self.MI_learner.parameters())
                self.optimizer = torch.optim.Adam( param_list_MI_learner, lr=self.args.lr, weight_decay=self.args.l2_coef)
                
                lr_lambda = lambda step: 1 if step < self.args.init_lr_num else self.args.lr_decay ** ((step - self.args.init_lr_num) // self.args.lr_decay_interval)
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

            if self.args.demonstration:
                prompt_format = "retrieve-demonstration-prompt"
            else:
                prompt_format = "retrieve-prompt"
        else:
            if self.args.demonstration:
                prompt_format = "general-demonstration-prompt"
            else:
                prompt_format = "general-prompt"
        self.print_logger.info(f"prompt_format: {prompt_format} \n")    
            
        with open(self.args.prompt_file, "r") as f:
            prompt_text = json.load(f)[prompt_format]

        self.prompt = PromptTemplate.from_template(prompt_text)
        if args.LLM != "chatGPT":
            self.pipe = pipeline(
                    "text-generation",
                    model=LLM,
                    tokenizer=self.LLM_tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    device_map="auto",
                    output_scores=True, 
                    return_dict_in_generate=True ) 
        else:
            self.pipe = LLMChain(prompt=self.prompt, llm=LLM)

    def random_select_demonstration(self, data_loader, batch_size):
        demon_prompt_list = []
        for i in range(batch_size):
            demon_prompt = ""
            for index, item in enumerate(data_loader):
                if self.args.demons_cnt < index+1:
                    break
                demon_prompt += "Demonstration " + str(index)+"\n Question: {} \n Options: {} \n Answer: <{}> \n\n".format(item["question"][0], item["options"][0], item["answer"][0])
            demon_prompt_list.append(demon_prompt)
        return demon_prompt_list

    def return_input_dict(self, dev_data_loader, data_item, retrieve_docs):
        if self.args.dataset == "OTTQA":
            if self.args.if_RA:
                if self.args.demonstration:
                    demonstration = self.random_select_demonstration(dev_data_loader, self.args.test_batch_size)
                    input_dict = {'question': data_item["question"], "context": retrieve_docs, "demonstration": demonstration}
                else:
                    input_dict = {'question': data_item["question"], "context": retrieve_docs}
            else:
                if self.args.demonstration:
                    demonstration = self.random_select_demonstration(dev_data_loader, self.args.test_batch_size)
                    input_dict = {'question': data_item["question"], "demonstration": demonstration}
                else:
                    input_dict = {'question': data_item["question"], }
        else:
            if self.args.if_RA:
                if self.args.demonstration:
                    demonstration = self.random_select_demonstration(dev_data_loader, self.args.test_batch_size)
                    input_dict = {'question': data_item["question"], 'options': data_item["options"], "context": retrieve_docs, "demonstration": demonstration}
                else:
                    input_dict = {'question': data_item["question"], 'options': data_item["options"], "context": retrieve_docs}
            else:
                if self.args.demonstration:
                    demonstration = self.random_select_demonstration(dev_data_loader, self.args.test_batch_size)
                    input_dict = {'question': data_item["question"], 'options': data_item["options"],  "demonstration": demonstration}
                else:
                    input_dict = {'question': data_item["question"], 'options': data_item["options"]}
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

    def train_proc(self, train_data_loader, dev_data_loader, test_data_loader):
        if (not self.args.if_RA) or (not self.args.if_MI_RA):
            raise Exception("need retrieve ! ")

        self.print_logger.info("Start training ... \n ")
        
        total_batch = len(train_data_loader)
        step_num = -1
        best_step = 0
        best_performce = 0
        eval_num = 0
        for epoch_num in range(self.args.epoch):
            for data_item in train_data_loader:
                self.MI_learner.train()
                step_num+=1
                question = data_item['question']
                labels = data_item['label']
                one_hot_labels = data_item['one_hot_label']
                batch_answer = data_item["answer"]

                query = self.get_multi_query_input(question, data_item)
                query_emb_list, attention_mask_list, bags_list = self.retriever.search_document(query) 
                retrieve_docs = combine_doc(bags_list)

                input_dict = self.return_input_dict(dev_data_loader, data_item, retrieve_docs)
                with torch.no_grad():
                    _, _,  _, _, batch_loss, batch_logit_log_softmax = self.pipeline_inference(input_dict, labels, batch_answer, training_flag=True, record_flag=False)

                loss_list, new_retrieve_docs, _ = self.MI_learner(query_emb_list, attention_mask_list, bags_list, batch_logit_log_softmax, one_hot_labels, batch_loss, self.retriever, True)
                total_loss = loss_list[-1]

                total_loss.backward()
                # new
                old_doc_len =  sum([len(i) for i in self.LLM_tokenizer(retrieve_docs)["input_ids"]]) / len(retrieve_docs)
                new_doc_len =  sum([len(i) for i in self.LLM_tokenizer(new_retrieve_docs)["input_ids"]]) / len(retrieve_docs)

                self.writer.add_scalar('Loss/total_loss', round(float(total_loss), 4), step_num)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], step_num)

                self.print_logger.info(f"epoch_num: {epoch_num}, training process num: {step_num}/{total_batch}, mse_loss: {round(float(loss_list[0]), 4)}, kl_soft_loss: {round(float(loss_list[1]), 4)}, kl_hard_loss: {round(float(loss_list[2]), 4)}, old_doc_len:{old_doc_len}, new_doc_len:{new_doc_len}, best_step:{best_step}, best_performce: {best_performce}")
                                       
                if (step_num + 1) % self.args.accumulation_steps == 0:
                    self.optimizer.step()
                    # if self.optimizer.param_groups[0]['lr'] >= 1e-5:
                        # self.scheduler.step()
                    self.optimizer.zero_grad()

                if (step_num % self.args.train_eval==0) and step_num>1:
                # if (step_num % self.args.train_eval==0) :
                    eval_num +=1
                    self.train_result_logger = empty_logger_file(self.train_result_logger)

                    break_cnt = 2 if self.args.test_code_flag else None
                    with torch.no_grad():
                        self.MI_learner.eval()
                        test_performce = self.test_proc(test_data_loader, dev_data_loader, step_num, break_cnt=break_cnt)
                        self.MI_learner.train()

                    if test_performce>best_performce:
                        best_performce = test_performce
                        best_step = step_num

                        # if step_num>10:
                        #     torch.save(self.MI_learner.state_dict(), self.args.dir_path+'/MI_' +str(best_performce)+'.pkl') 

    

    def test_proc(self, test_data_loader, dev_data_loader, eval_num=0, break_cnt=None):
            
        self.print_logger.info("\n Start test ...  ")
        start_time = time.time()

        all_test_labels = []
        all_test_prediction_ids = []
        all_test_predictions = []
        all_test_answers = []
        old_doc_len = 0
        new_doc_len = 0
        total_hallucination_cnt = 0

        for index, data_item in enumerate(test_data_loader):
            if index%200==0:
                self.print_logger.info(f"testing process num: {index}")
            question = data_item['question']
            batch_label = data_item["label"]
            batch_answer = data_item["answer"]

            if self.args.if_RA:
                query = self.get_multi_query_input(question, data_item)

                query_emb_list, attention_mask_list, bags_list = self.retriever.search_document(query) 
                retrieve_docs = combine_doc(bags_list)

                if self.args.infer_add_gold_retrieval:
                    retrieve_docs = self.add_gold_retrieval(retrieve_docs, data_item)

                old_doc_len +=  sum([len(i) for i in self.LLM_tokenizer(retrieve_docs)["input_ids"]]) / len(retrieve_docs)
                if self.args.if_MI_RA:
                    _, retrieve_docs, _ = self.MI_learner(query_emb_list, attention_mask_list, bags_list, "batch_logit_log_softmax", "one_hot_labels", "batch_loss", self.retriever, False)
                    new_doc_len +=  sum([len(i) for i in self.LLM_tokenizer(retrieve_docs)["input_ids"]]) / len(retrieve_docs)
            else:
                retrieve_docs = ""

            input_dict = self.return_input_dict(dev_data_loader, data_item, retrieve_docs)
            batch_pred, batch_id_pred, batch_hallucination_cnt, _, _, _ = self.pipeline_inference(input_dict, batch_label, batch_answer, training_flag=False, record_flag=True)
            total_hallucination_cnt+=batch_hallucination_cnt

            all_test_labels+=batch_label
            all_test_prediction_ids+=batch_id_pred

            all_test_predictions+=batch_pred
            all_test_answers+=batch_answer

            break_cnt = 2 if self.args.test_code_flag else None
            if break_cnt is not None and break_cnt<index:
                break
        
        cost_time  = (time.time() - start_time)/60
        old_doc_len = old_doc_len / len(test_data_loader)   
        new_doc_len = new_doc_len / len(test_data_loader)   

        if self.args.dataset == "OTTQA":
            test_f1 , test_EM =  self.my_metrics.get_raw_scores(all_test_predictions, all_test_answers)   
            
            self.args.print_logger.info(f"test: f1 {test_f1}, test: EM {test_EM}, old_doc_len:{old_doc_len}, new_doc_len:{new_doc_len} \n ")
            record_performance = test_f1

            self.writer.add_scalar('Performance/test/EM', test_EM, eval_num )
            self.writer.add_scalar('Performance/test/f1', test_f1, eval_num )
        else:
            test_acc, test_precision, test_recall, test_f1 = self.my_metrics.acc_PRF(all_test_labels, all_test_prediction_ids)
            self.args.print_logger.info(f"test: acc {test_acc}, f1 {test_f1}, precision {test_precision}, recall {test_recall}, old_doc_len:{old_doc_len}, new_doc_len:{new_doc_len}, hallucination: {total_hallucination_cnt/len(test_data_loader)/len(question)} ")
            self.args.print_logger.info(f"cost_time: {cost_time} \n ")
            record_performance = test_acc

            self.writer.add_scalar('Performance/test/acc', test_acc, eval_num )
            self.writer.add_scalar('Performance/test/precision', test_precision, eval_num )
            self.writer.add_scalar('Performance/test/recall', test_recall, eval_num )
            self.writer.add_scalar('Performance/test/f1', test_f1, eval_num )

        return record_performance
         
    def pipeline_inference(self, input_dict, label, batch_answer, training_flag=False, record_flag=True):
        if self.args.LLM == "chatGPT":
            batch_pred, batch_id_pred, batch_hallucination_cnt, save_doc_num = self.non_local_llm_infer(input_dict, label, batch_answer, training_flag, record_flag)
            batch_loss, batch_logit_log_softmax = 0, 0
        else:
            batch_pred, batch_id_pred, batch_hallucination_cnt, save_doc_num, batch_loss, batch_logit_log_softmax= self.local_llm_infer(input_dict, label, batch_answer, training_flag, record_flag)

        return batch_pred, batch_id_pred, batch_hallucination_cnt, save_doc_num, batch_loss, batch_logit_log_softmax

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
            pred, id_pred, hallucination_cnt = self.pasrse_record_res(self.prompt.format(**current_inputs) , label[index2], pred, batch_answer[index2], training_flag, record_flag) 
            batch_pred.append(pred)  
            batch_id_pred.append(id_pred)
            batch_hallucination_cnt+=hallucination_cnt

        return batch_pred, batch_id_pred, batch_hallucination_cnt, save_doc_num
    
    def local_llm_infer(self, input_dict, label, batch_answer, training_flag=False, record_flag=True):
        save_doc_num = [self.args.n_docs]*len(label)
       
    
        my_input_list = []
        keys = input_dict.keys()
        for values in zip(*input_dict.values()):
            current_inputs = dict(zip(keys, values))
            my_input = self.prompt.format(**current_inputs)
            my_input_list.append(my_input)
        
        batch_id_pred = []
        batch_pred = []
        batch_hallucination_cnt = 0

        inputs = self.LLM_tokenizer(my_input_list, return_tensors="pt", padding=True).to(self.args.device)
        outputs = self.LLM.generate(**inputs, max_new_tokens=self.args.max_new_tokens, 
                                    num_return_sequences=1, 
                                    temperature=self.args.temperature,
                                    top_p=self.args.top_p,
                                    return_dict_in_generate=True, 
                                    output_scores=True,
                                    output_hidden_states=True,
                                    do_sample=True
                                    # length_penalty=self.args.length_penalty,
                                    # num_beams=self.args.num_beams,
                                )
        logit_log_softmax, batch_loss = self.get_logits_and_loss(outputs, label)

        for index, (input, output, answer) in enumerate(zip(inputs["input_ids"], outputs["sequences"], batch_answer)):
            generation = self.LLM_tokenizer.decode(output, skip_special_tokens=True)
            pred, id_pred, hallucination_cnt = self.pasrse_record_res(my_input_list[index], label[index], generation, answer, training_flag, record_flag)
            batch_pred.append(pred)
            batch_id_pred.append(id_pred)
            batch_hallucination_cnt+=hallucination_cnt

        return batch_pred, batch_id_pred, batch_hallucination_cnt, save_doc_num, batch_loss, logit_log_softmax

    def get_logits_and_loss(self, outputs, label):
        last_hidden_states = outputs["hidden_states"][0][-1]
        logit = self.LLM.lm_head(last_hidden_states)[:, -1, :]
        logit_log_softmax = F.log_softmax(logit, dim=-1)

        label = torch.LongTensor(label).to(logit_log_softmax.device)
        loss_fct = nn.NLLLoss(reduction="none")
        batch_loss = loss_fct(logit_log_softmax, label.view(-1))

        return logit_log_softmax, batch_loss
    
    def pasrse_record_res(self, my_input, label, generation, answer, training_flag, record_flag):
        pred, id_pred, hallucination_cnt = extracted_token_id_label(generation, label, self.LLM_tokenizer, self.args.dataset, self.prompt, self.args.LLM)

        if training_flag:
            result_logger = self.train_result_logger
        else:    
            result_logger = self.test_result_logger

        if record_flag:
            result_logger.info(f"my_input: {my_input}")
            result_logger.info(f"answer:   {answer} ")
            result_logger.info(f"pred:   {pred} ")
            result_logger.info(f"=================================================================================================================================================================================================\n\n")
            # result_logger.info(f"label:   {[self.LLM_tokenizer._convert_id_to_token(int(label_i))   for label_i in label] } ")
            # result_logger.info(f"id_pred: {[self.LLM_tokenizer._convert_id_to_token(id_pred_i) for id_pred_i in id_pred] } "+ "\n========================================================================================================================")
        return pred, id_pred, hallucination_cnt
    

   