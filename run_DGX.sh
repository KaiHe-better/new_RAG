# llama2-7b_USMLE
nohup python run.py --ID USMLE_00  --gpu 4 --config llama2-7b_USMLE.yaml --dataset USMLE  >/dev/null 2>&1 &  
2024-05-04 13:40:54,956 test: acc 31.97, f1 25.77, precision 30.72, recall 34.55, old_doc_len:0.0, new_doc_len:0.0, hallucination: 0.0 

# llama2-7b_USMLE_RA
nohup python run.py --ID USMLE_0  --gpu 7 --config llama2-7b_USMLE_RA.yaml --dataset USMLE  >/dev/null 2>&1 &  
2024-05-04 10:48:32,949 test: acc 38.18, f1 38.1, precision 38.18, recall 38.58, old_doc_len:1253.7511773940346, new_doc_len:0.0, hallucination: 0.0 


# llama2-7b_USMLE_MI_RA
nohup python run.py --ID USMLE_06_     --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE  >/dev/null 2>&1 &  
test: acc 37.86, f1 37.78, precision 37.87, recall 38.21, old_doc_len:1253.7511773940346, new_doc_len:827.265306122449

nohup python run.py --ID USMLE_07_Loss --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 500 >/dev/null 2>&1 &
问题:保留了所有的检索结果
2024-05-04 09:42:01,634 test: acc 38.18, f1 38.1, precision 38.18, recall 38.58, old_doc_len:1253.7511773940346, new_doc_len:1253.256671899529, hallucination: 0.0 


nohup python run.py --ID USMLE_0 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+len_penalty --len_penalty_weight 100 >/dev/null 2>&1 &
nohup python run.py --ID USMLE_1 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_hard+len_penalty --len_penalty_weight 100 >/dev/null 2>&1 &
nohup python run.py --ID USMLE_2 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1     --len_penalty_weight 100 >/dev/null 2>&1 &
nohup python run.py --ID USMLE_2 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+kl_hard+len_penalty --soft_weight 0.7 --hard_weight 0.3 --len_penalty_weight 100 >/dev/null 2>&1 &
