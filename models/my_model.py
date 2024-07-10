# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, MultiheadAttention, Linear, Dropout, LayerNorm, TransformerEncoder
import torch.nn.functional as F
from utils.utils import combine_doc
from sklearn.metrics import  accuracy_score
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.metrics import compute_exact

class My_gate(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)
        self.args = args

        encoder_layer_gate = nn.TransformerEncoderLayer(d_model=3, nhead=1, dropout=0, batch_first=True).to(self.args.device)
        self.trans_gate = nn.TransformerEncoder(encoder_layer_gate, num_layers=1)
        self.gate_linear = Linear(3, 2).to(self.args.device)
        self.gate_loss = nn.CrossEntropyLoss(weight=torch.tensor([args.gate_weight_0, args.gate_weight_1], dtype=torch.float32).to(args.device))

    def get_new_close_lable(self, general_batch_pred, batch_pred, batch_answer):
        new_label_list = []
        label_0_0=0
        label_0_1=0
        label_1_0=0
        label_1_1=0
        for general_batch_pred_item, batch_pred_item, batch_answer_item in zip(general_batch_pred, batch_pred, batch_answer):
            if (batch_answer_item != general_batch_pred_item) and (batch_answer_item != batch_pred_item) :
                new_label_list.append(1)
                label_0_0+=1

            if (batch_answer_item != general_batch_pred_item) and (batch_answer_item == batch_pred_item) :
                new_label_list.append(1)
                label_0_1+=1

            if (batch_answer_item == general_batch_pred_item) and  (batch_answer_item != batch_pred_item) :
                new_label_list.append(0)
                label_1_0+=1

            if (batch_answer_item == general_batch_pred_item) and (batch_answer_item == batch_pred_item):
                new_label_list.append(0)
                label_1_1+=1

        return new_label_list, [label_0_0, label_0_1, label_1_0, label_1_1]

    def get_new_open_lable(self, general_batch_pred, batch_pred, batch_answer):
        general_batch_label_list = []
        batch_label_list = []
        for general_pred, pred, label in zip(general_batch_pred, batch_pred, batch_answer):
            pred = pred.split(". \n")[0] 
            general_pred = general_pred.split(". \n")[0] 

            general_batch_label_list.append(compute_exact(label, general_pred)) 
            batch_label_list.append(compute_exact(label, pred)) 


        new_label_list = []
        label_0_0=0
        label_0_1=0
        label_1_0=0
        label_1_1=0
        for general_batch_label_item, batch_label_item in zip(general_batch_label_list, batch_label_list):
            if general_batch_label_item==0 and batch_label_item ==0 :
                new_label_list.append(1)
                label_0_0+=1

            if general_batch_label_item==0 and batch_label_item ==1 :
                new_label_list.append(1)
                label_0_1+=1

            if general_batch_label_item==1 and  batch_label_item ==0 :
                new_label_list.append(0)
                label_1_0+=1

            if general_batch_label_item==1 and batch_label_item ==1:
                new_label_list.append(0)
                label_1_1+=1

        return new_label_list, [label_0_0, label_0_1, label_1_0, label_1_1]

    def make_gate_input(self, batch_loss, general_batch_loss, batch_logit_log_softmax, general_batch_logit_log_softmax, raw_ques_emb_list, raw_doc_emb_list):
        batch_logit_log_softmax = -torch.max(batch_logit_log_softmax, dim=-1)[0]
        general_batch_logit_log_softmax = -torch.max(general_batch_logit_log_softmax, dim=-1)[0]
        gate_input = torch.cat((general_batch_loss, batch_logit_log_softmax, general_batch_logit_log_softmax), dim=-1)
        return gate_input
    
    def forward(self, general_batch_pred, batch_pred, batch_answer, batch_loss, 
                  raw_ques_emb_list, raw_doc_emb_list, general_batch_loss, 
                  batch_logit_log_softmax, general_batch_logit_log_softmax):
        
        gate_input = self.make_gate_input(batch_loss, general_batch_loss, batch_logit_log_softmax, general_batch_logit_log_softmax, raw_ques_emb_list, raw_doc_emb_list)
        gate_logit = self.trans_gate(gate_input)
        gate_logit = self.gate_linear(gate_logit)
        gate_res = torch.argmax(gate_logit, dim=-1)
        
        if batch_pred is not None:
            if self.args.dataset in ["USMLE", "MedMCQA", "HEADQA"]:
                new_lable, new_label_count_list = self.get_new_close_lable(general_batch_pred, batch_pred, batch_answer)
            else:
                new_lable, new_label_count_list = self.get_new_open_lable(general_batch_pred, batch_pred, batch_answer)

            gate_loss = self.gate_loss(gate_logit, torch.LongTensor(new_lable).to(gate_logit.device) )

        else:
            gate_loss = 0

        return gate_loss, gate_res, new_lable, new_label_count_list
    

class My_MI_learner(nn.Module):
    def __init__(self, args, vocab_size):
        nn.Module.__init__(self)
        self.args = args

        encoder_layer_ques = nn.TransformerEncoderLayer(d_model=self.args.d_model, nhead=self.args.nhead, dropout=self.args.dropout, batch_first=True).to(self.args.device)
        self.trans_ques = nn.TransformerEncoder(encoder_layer_ques, num_layers=2)

        encoder_layer_doc = nn.TransformerEncoderLayer(d_model=self.args.d_model, nhead=self.args.nhead, dropout=self.args.dropout, batch_first=True).to(self.args.device)
        self.trans_doc = nn.TransformerEncoder(encoder_layer_doc, num_layers=2)

        self.multi_head_ques = MultiLayerCrossAttention(args, num_layers=self.args.num_layers).to(self.args.device)
        self.multi_head_doc = MultiLayerCrossAttention(args, num_layers=self.args.num_layers).to(self.args.device)

        # new change
        self.linear_kl = Linear(self.args.d_model*2, vocab_size).to(self.args.device)

        self.len_loss_function = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.kl_loss_hard = nn.NLLLoss()

    def return_hierarchical_bag(self, bag, text_splitter):
        new_bag_list =[]
        new_doc_list =[]
        for item in bag:
            new_doc_list.append(Document(page_content=item))
        chunks = text_splitter.split_documents(new_doc_list)
        new_bag_list = [i.page_content for i in chunks]
        return new_bag_list

    def forward(self, bags_list, query_emb, ques_att_masks, retriever, train_flag, one_hot_labels, batch_logit_log_softmax):
        total_kl_logit = []
        select_doc = []
        select_doc_num = []
        att_weights_list = []
        raw_ques_emb_list = []
        raw_doc_emb_list = []
        for bag, raw_ques_emb, ques_att_mask in zip(bags_list, query_emb, ques_att_masks):
            if self.args.if_hierarchical_retrieval:
                text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(retriever.tokenizer, 
                                                                                chunk_size=int(self.args.chunk_size/self.args.hierarchical_ratio), 
                                                                                chunk_overlap=int(self.args.chunk_overlap/self.args.hierarchical_ratio))
                bag = self.return_hierarchical_bag(bag, text_splitter)

            with torch.no_grad():
                raw_doc_emb, _, doc_att_mask = retriever.embed_queries(self.args, bag)

            raw_ques_emb_list.append(raw_ques_emb[0])
            raw_doc_emb_list.append(torch.mean(raw_doc_emb[:, 0, :], dim=0))

            ques_att_mask =(1- ques_att_mask).bool() 
            doc_att_mask = (1- doc_att_mask).bool()
            raw_ques_emb = self.trans_ques(raw_ques_emb, src_key_padding_mask=ques_att_mask)[0, :].unsqueeze(0).unsqueeze(0)
            raw_doc_emb  = self.trans_doc(raw_doc_emb, src_key_padding_mask=doc_att_mask)[:, 0, :].unsqueeze(0)

            que_emb, att_weights  = self.multi_head_ques(raw_ques_emb, raw_doc_emb)  # raw_ques_emb (1,1,768)  raw_doc_emb (1,24,768) att_weights (1,1,24)
            att_weights_list.append(att_weights)

            doc_emb, _  = self.multi_head_doc(raw_doc_emb, raw_ques_emb) # doc_emb (1,24,768)

            select_index = torch.where( att_weights.squeeze() >= 1/len(bag)* self.args.quantile_num )[0]
            select_doc.append( combine_doc([[bag[i] for i in select_index ]])[0] )
            select_doc_num.append(len(select_index))
           
            if train_flag:
                doc_emb = torch.mean(doc_emb, dim=1).squeeze()
                # new change
                doc_emb = torch.cat((doc_emb, que_emb.squeeze()), dim=0)

                if "kl" in self.args.loss_list:
                    kl_logit = self.linear_kl(doc_emb) 
                    MI_logit_log_softmax = F.log_softmax(kl_logit, dim=0)
                    total_kl_logit.append(MI_logit_log_softmax)

        total_loss = 0
        len_penalty_loss = 0
        kl_soft_loss = 0
        kl_hard_loss = 0
        if train_flag:
            if "len_penalty" in self.args.loss_list:
                for item in att_weights_list:
                    len_penalty_label = (torch.ones(item.size())*1/len(bags_list[0])).to(item.device)
                    len_penalty_loss += self.len_loss_function(item , len_penalty_label) 
                len_penalty_loss = self.args.len_penalty_weight * len_penalty_loss

            if "kl_soft" in self.args.loss_list:
                kl_soft_loss = self.kl_loss(torch.stack(total_kl_logit), batch_logit_log_softmax.to(MI_logit_log_softmax.device)) 
                kl_soft_loss = self.args.soft_weight * kl_soft_loss

            if "kl_hard" in self.args.loss_list:
                label = torch.argmax(one_hot_labels, dim=1).to(self.args.device)
                kl_hard_loss = self.kl_loss_hard(torch.stack(total_kl_logit), label) 
                kl_hard_loss = self.args.hard_weight * kl_hard_loss

        total_loss = kl_soft_loss+kl_hard_loss+len_penalty_loss

        raw_ques_emb_list = torch.stack(raw_ques_emb_list)
        raw_doc_emb_list  = torch.stack(raw_doc_emb_list)

        return [len_penalty_loss, kl_soft_loss, kl_hard_loss, total_loss], select_doc, select_doc_num, raw_ques_emb_list, raw_doc_emb_list



class CrossAttentionLayer(nn.Module):
    def __init__(self, args):
        super(CrossAttentionLayer, self).__init__()
        self.args = args
        self.attention = nn.MultiheadAttention(embed_dim=self.args.d_model, 
                                               num_heads=self.args.nhead, 
                                               dropout=self.args.dropout, 
                                               batch_first=True)


    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, attn_wieght = self.attention(query, key, value, key_padding_mask=key_padding_mask, average_attn_weights=True)
        return attn_output, attn_wieght

class MultiLayerCrossAttention(nn.Module):
    def __init__(self, args, num_layers):
        super(MultiLayerCrossAttention, self).__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(args) for _ in range(num_layers)
        ])

    def forward(self, query, k_v, key_padding_mask=None):
        for layer in self.layers:
            query, attn_wieght = layer(query, k_v, k_v, key_padding_mask=key_padding_mask)
        return query, attn_wieght

    



