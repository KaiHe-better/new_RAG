import torch
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
import json
import re
import collections
import string
import sys



def acc_PRF(  pred, label):

    acc = accuracy_score(label, pred)
    precision = precision_score(label, pred, average="macro")
    recall = recall_score(label, pred, average="macro")
    f1 = f1_score(label, pred, average="macro")

    acc = round(acc*100, 2)
    precision = round(precision*100, 2)
    recall = round(recall*100, 2)
    f1 = round(f1*100, 2)

    return acc, precision, recall, f1

def normalize_answer(  s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(  s):
    if not s:
        return []
    return  normalize_answer(s).split()

def compute_exact( a_gold_list, a_pred):
    res_list = []
    a_pred =  normalize_answer(a_pred)
    for a_gold in a_gold_list:
        # res = int(a_pred ==  normalize_answer(a_gold) )
        res = int(normalize_answer(a_gold) in a_pred)
        if res ==1:
            return 1
    return res

def compute_f1(  a_gold_list, a_pred):
    f1_list = []
    for a_gold in a_gold_list:
        gold_toks =  get_tokens(a_gold)
        pred_toks =  get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            f1_list.append(int(gold_toks == pred_toks))
            continue
        if num_same == 0:
            f1_list.append(0)
            continue

        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)

        f1_list.append(f1)
    return max(f1_list)

def F_EM(  reference, examples):
    """
    reference = label,  examples = preds
    """
    exact_scores_list = []
    f1_scores_list = []
    
    for pred, label in zip(examples, reference):
        pred = pred.split(". \n")[0] 
        exact_scores = compute_exact(label, pred)
        f1_scores = compute_f1(label, pred) 
        
        exact_scores_list.append(exact_scores)
        f1_scores_list.append(f1_scores)

    f1 = round(sum(f1_scores_list)/len(f1_scores_list)*100, 2)
    exact_scores = round(sum(exact_scores_list)/len(exact_scores_list)*100, 2)
    return f1, exact_scores



