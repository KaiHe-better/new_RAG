# llama2-7b_USMLE
nohup python run.py --ID USMLE_00  --gpu 4 --config llama2-7b_USMLE.yaml --dataset USMLE  >/dev/null 2>&1 &  
2024-05-04 13:40:54,956 test: acc 31.97, f1 25.77, precision 30.72, recall 34.55, old_doc_len:0.0, new_doc_len:0.0, hallucination: 0.0 



# llama2-7b_USMLE_RA
nohup python run.py --ID USMLE_0  --gpu 7 --config llama2-7b_USMLE_RA.yaml --dataset USMLE  >/dev/null 2>&1 &  
2024-05-04 10:48:32,949 test: acc 38.18, f1 38.1, precision 38.18, recall 38.58, old_doc_len:1253.7511773940346, new_doc_len:0.0, hallucination: 0.0 

nohup python run.py --ID USMLE_0  --gpu 7 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --n_docs 10 >/dev/null 2>&1 &  
2024-05-06 11:23:41,434 test: acc 35.9, f1 35.82, precision 35.91, recall 36.86, old_doc_len:2481.970957613815, new_doc_len:0.0, hallucination: 1.1 



# llama2-7b_USMLE_MI_RA  --n_docs 5
nohup python run.py --ID USMLE_06_     --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE  >/dev/null 2>&1 &  
test: acc 37.86, f1 37.78, precision 37.87, recall 38.21, old_doc_len:1253.7511773940346, new_doc_len:827.265306122449

nohup python run.py --ID USMLE_07_Loss --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 500 >/dev/null 2>&1 &
问题:保留了所有的检索结果
2024-05-04 09:42:01,634 test: acc 38.18, f1 38.1, precision 38.18, recall 38.58, old_doc_len:1253.7511773940346, new_doc_len:1253.256671899529, hallucination: 0.0 

nohup python run.py --ID USMLE_0 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+len_penalty --len_penalty_weight 100 >/dev/null 2>&1 &
问题:保留了所有的检索结果
2024-05-05 01:34:19,214 test: acc 38.41, f1 38.31, precision 38.34, recall 38.8, old_doc_len:1253.7511773940346, new_doc_len:1214.9207221350077, hallucination: 0.0 

nohup python run.py --ID USMLE_1 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_hard+len_penalty --len_penalty_weight 100 >/dev/null 2>&1 &
问题:保留了所有的检索结果
2024-05-05 01:44:45,764 test: acc 38.26, f1 38.16, precision 38.23, recall 38.63, old_doc_len:1253.7511773940346, new_doc_len:1227.541601255887, hallucination: 0.0 

nohup python run.py --ID USMLE_2 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1     --len_penalty_weight 100 >/dev/null 2>&1 &
问题:保留了所有的检索结果
2024-05-05 06:38:47,023 test: acc 38.41, f1 38.33, precision 38.42, recall 38.8, old_doc_len:1253.7511773940346, new_doc_len:1247.513343799058, hallucination: 0.0 

nohup python run.py --ID USMLE_2 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+kl_hard+len_penalty --soft_weight 0.7 --hard_weight 0.3 --len_penalty_weight 100 >/dev/null 2>&1 &
问题:保留了所有的检索结果
2024-05-05 05:17:11,951 test: acc 38.33, f1 38.26, precision 38.36, recall 38.74, old_doc_len:1253.7511773940346, new_doc_len:1240.298273155416, hallucination: 0.0 

nohup python run.py --ID USMLE_0 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+len_penalty --len_penalty_weight 10 >/dev/null 2>&1 &
问题:保留的太多了
2024-05-06 01:00:12,807 test: acc 38.33, f1 38.23, precision 38.25, recall 38.7, old_doc_len:1253.7511773940346, new_doc_len:1123.4356357927786, hallucination: 0.0 

nohup python run.py --ID USMLE_1 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_hard+len_penalty --len_penalty_weight 10 >/dev/null 2>&1 &
问题:保留的太多了
2024-05-06 04:03:40,874 test: acc 38.33, f1 38.26, precision 38.33, recall 38.73, old_doc_len:1253.7511773940346, new_doc_len:1235.861852433281, hallucination: 0.0 

nohup python run.py --ID USMLE_2 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1     --len_penalty_weight 10 >/dev/null 2>&1 &
问题:保留的太多了
2024-05-06 04:00:10,466 test: acc 38.57, f1 38.52, precision 38.64, recall 39.07, old_doc_len:1253.7511773940346, new_doc_len:1193.9246467817895, hallucination: 0.0 

nohup python run.py --ID USMLE_3 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+kl_hard+len_penalty --soft_weight 0.7 --hard_weight 0.3 --len_penalty_weight 10 >/dev/null 2>&1 &
问题:保留的太多了
2024-05-06 04:47:22,311 test: acc 38.33, f1 38.25, precision 38.29, recall 38.67, old_doc_len:1253.7511773940346, new_doc_len:1191.190737833595, hallucination: 0.0 



# llama2-7b_USMLE_MI_RA  --n_docs 10

nohup python run.py --ID USMLE_0 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+len_penalty --len_penalty_weight 5 >/dev/null 2>&1 &
2024-05-06 15:51:21,175 test: acc 37.63, f1 37.57, precision 37.87, recall 38.57, old_doc_len:2481.970957613815, new_doc_len:1878.870486656201, hallucination: 0.0 

nohup python run.py --ID USMLE_1 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_hard+len_penalty --len_penalty_weight 5 >/dev/null 2>&1 &
2024-05-06 12:04:36,145 test: acc 36.84, f1 36.78, precision 37.02, recall 37.68, old_doc_len:2481.970957613815, new_doc_len:1896.5, hallucination: 0.08 

nohup python run.py --ID USMLE_2 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1     --len_penalty_weight 5 >/dev/null 2>&1 &
2024-05-07 10:06:44,816 test: acc 37.47, f1 37.42, precision 37.7, recall 38.49, old_doc_len:2481.970957613815, new_doc_len:2155.784929356358, hallucination: 0.08 

nohup python run.py --ID USMLE_3 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 0.7 --hard_weight 0.3 --len_penalty_weight 5 >/dev/null 2>&1 &
2024-05-06 15:32:48,852 test: acc 37.16, f1 37.09, precision 37.35, recall 37.9, old_doc_len:2481.970957613815, new_doc_len:1786.539246467818, hallucination: 0.08 




# new llama2-7b_USMLE_MI_RA  --n_docs 10

nohup python run.py --ID USMLE_0 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --gate_weight 10  --len_penalty_weight 4 >/dev/null 2>&1 &
nohup python run.py --ID USMLE_3 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --gate_weight 50  --len_penalty_weight 4 >/dev/null 2>&1 &

