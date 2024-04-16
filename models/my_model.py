# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, Linear
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from utils.utils import combine_doc
from sklearn.metrics import  accuracy_score
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class My_MI_learner(nn.Module):
    def __init__(self, args, vocab_size):
        nn.Module.__init__(self)
        self.args = args

        config = BertConfig(
                hidden_size=self.args.d_model,  
                num_attention_heads=self.args.nhead,  
                hidden_dropout_prob=self.args.dropout,  
                attention_probs_dropout_prob=self.args.dropout  )


        self.trans_ques = TransformerEncoderLayer(self.args.d_model, self.args.nhead, dropout=self.args.dropout, batch_first=True).to(self.args.device)
        self.trans_doc = TransformerEncoderLayer(self.args.d_model, self.args.nhead, dropout=self.args.dropout, batch_first=True).to(self.args.device)


        self.multi_head_ques = MultiLayerCrossAttention(args, num_layers=self.args.num_layers).to(self.args.device)
        self.multi_head_doc = MultiLayerCrossAttention(args, num_layers=self.args.num_layers).to(self.args.device)

        self.linear_kl = Linear(self.args.d_model*2, vocab_size).to(self.args.device)
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
    
    def forward(self, query_emb, ques_att_masks, bags_list, batch_logit_log_softmax, one_hot_labels, batch_loss, retriever, train_flag):
        total_mse_logit = []
        total_kl_logit = []
        select_doc = []
        select_doc_num = []
        for bag, raw_ques_emb, raw_ques_att_mask in zip(bags_list, query_emb, ques_att_masks):
            if self.args.if_hierarchical_retrieval:
                text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(retriever.tokenizer, 
                                                                                chunk_size=int(self.args.chunk_size/self.args.hierarchical_ratio), 
                                                                                chunk_overlap=int(self.args.chunk_overlap/self.args.hierarchical_ratio))
                bag = self.return_hierarchical_bag(bag, text_splitter)

            with torch.no_grad():
                raw_doc_emb, _, raw_doc_attention_mask = retriever.embed_queries(self.args, bag)

            ques_att_mask =(1- raw_ques_att_mask).bool() 
            doc_att_mask = (1- raw_doc_attention_mask).bool()
            raw_ques_emb = self.trans_ques(raw_ques_emb, src_key_padding_mask=ques_att_mask)
            raw_doc_emb  = self.trans_doc(raw_doc_emb, src_key_padding_mask=doc_att_mask)

            raw_ques_emb_mask = raw_ques_emb*raw_ques_att_mask.unsqueeze(-1).expand(-1, raw_ques_emb.size(-1))
            raw_doc_emb_mask  = raw_doc_emb*raw_doc_attention_mask.unsqueeze(-1).expand(-1, -1, raw_doc_emb.size(-1))                                                                
            raw_ques_emb = torch.mean(raw_ques_emb_mask, dim=0).unsqueeze(0)
            raw_doc_emb = torch.mean(raw_doc_emb_mask, dim=1)

            que_emb, att_weights  = self.multi_head_ques(raw_ques_emb, raw_doc_emb)
            doc_emb, _  = self.multi_head_doc(raw_doc_emb, raw_ques_emb)

            select_index = torch.where( att_weights.squeeze() >= 1/len(bag)* self.args.quantile_num )[0]
            select_doc = select_doc + combine_doc([[bag[i] for i in select_index ]]) 
            select_doc_num.append(len(select_index))
           
            if train_flag:
                doc_emb = torch.mean(doc_emb, dim=0).squeeze()
                # new change
                doc_emb = torch.cat((doc_emb, que_emb.squeeze()), dim=0)

                if "kl" in self.args.loss_list:
                    kl_logit = self.linear_kl(doc_emb) 
                    MI_logit_log_softmax = F.log_softmax(kl_logit, dim=0)
                    total_kl_logit.append(MI_logit_log_softmax)


        total_loss = 0
        mse_loss = 0
        kl_soft_loss = 0
        kl_hard_loss = 0
        if train_flag:

            if "kl_soft" in self.args.loss_list:
                kl_soft_loss = self.kl_loss(torch.stack(total_kl_logit), batch_logit_log_softmax.to(MI_logit_log_softmax.device)) 
                kl_soft_loss = self.args.soft_weight * kl_soft_loss

            if "kl_hard" in self.args.loss_list:
                label = torch.argmax(one_hot_labels, dim=1).to(self.args.device)
                kl_hard_loss = self.kl_loss_hard(torch.stack(total_kl_logit), label) 
                kl_hard_loss = self.args.hard_weight * kl_hard_loss

        total_loss = mse_loss+kl_soft_loss+kl_hard_loss
        return  [mse_loss, kl_soft_loss, kl_hard_loss, total_loss], select_doc, select_doc_num

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

    



