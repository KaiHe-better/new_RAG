# llama2-7b_USMLE
nohup python run.py --ID USMLE_00  --gpu 4 --config llama2-7b_USMLE.yaml --dataset USMLE  >/dev/null 2>&1 &  
2024-05-04 13:40:54,956 test: acc 31.97, f1 25.77, precision 30.72, recall 34.55, old_doc_len:0.0, new_doc_len:0.0, hallucination: 0.0 

# llama2-7b_USMLE_RA
nohup python run.py --ID USMLE_0  --gpu 5 --config llama2-7b_USMLE_RA.yaml --dataset USMLE  >/dev/null 2>&1 &  
2024-05-04 10:48:32,949 test: acc 38.18, f1 38.1, precision 38.18, recall 38.58, old_doc_len:1253.7511773940346, new_doc_len:0.0, hallucination: 0.0 


# llama2-7b_USMLE_MI_RA
nohup python run.py --ID USMLE_06     --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE  >/dev/null 2>&1 &  
test: acc 37.86, f1 37.78, precision 37.87, recall 38.21, old_doc_len:1253.7511773940346, new_doc_len:827.265306122449

nohup python run.py --ID USMLE_07_Loss --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 500 >/dev/null 2>&1 &
问题:保留了所有的检索结果
2024-05-04 09:42:01,634 test: acc 38.18, f1 38.1, precision 38.18, recall 38.58, old_doc_len:1253.7511773940346, new_doc_len:1253.256671899529, hallucination: 0.0 

===================================================================================================================================================================================================

nohup python run.py --ID USMLE_0 --gpu 4 --config llama3-8b_USMLE_RA.yaml --dataset USMLE --n_docs 5   >/dev/null 2>&1 &
2024-05-12 19:57:45,893 test: acc 53.73, f1 53.62, precision 53.79, recall 54.13, old_doc_len:1021.0675039246468, new_doc_len:1021.0675039246468, hallucination: 10.68 

nohup python run.py --ID USMLE_01 --gpu 7 --config llama3-8b_USMLE_RA.yaml --dataset USMLE --n_docs 10  >/dev/null 2>&1 &
2024-05-12 20:01:40,753 test: acc 59.86, f1 59.69, precision 59.88, recall 59.95, old_doc_len:2020.9819466248039, new_doc_len:2020.9819466248039, hallucination: 0.31 



nohup python run.py --ID USMLE_2 --gpu 7 --config llama3-8b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 5  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 10 >/dev/null 2>&1 &
2024-05-13 01:25:04,476 test: acc 64.89, f1 64.67, precision 64.93, recall 64.97, old_doc_len:1021.0675039246468, new_doc_len:190.43720565149135, hallucination: 5.97 

nohup python run.py --ID USMLE_3 --gpu 4 --config llama3-8b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 10 >/dev/null 2>&1 &
2024-05-15 09:42:19,157 test: acc 66.93, f1 66.69, precision 66.9, recall 66.79, old_doc_len:2020.9819466248039, new_doc_len:381.6773940345369, hallucination: 0.39 


# 这两个训练时间太长，这2个貌似没区别， 000比001稍好
nohup python run.py --ID USMLE_000 --gpu 4 --config llama3-8b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 3 >/dev/null 2>&1 &
2024-05-23 08:18:46,111 test: acc 67.48, f1 67.26, precision 67.48, recall 67.33, old_doc_len:2020.9819466248039, new_doc_len:290.8006279434851, hallucination: 1.73 
67.01

nohup python run.py --ID USMLE_001 --gpu 5 --config llama3-8b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard >/dev/null 2>&1 &
2024-05-23 04:23:07,022 test: acc 67.48, f1 67.25, precision 67.45, recall 67.32, old_doc_len:2020.9819466248039, new_doc_len:347.8689167974882, hallucination: 0.39 
67.01

===================================================================================================================================================================================================
# USMLE
nohup  python run.py --ID USMLE_5 --gpu 6 --RA_method No_RA   --dataset USMLE --n_docs 10    >/dev/null 2>&1 &
2024-06-17 22:57:06,374 test: acc 58.99, f1 58.79, precision 59.05, recall 59.42, old_doc_len:0.0, new_doc_len:0.0, hallucination: 2.44 

nohup  python run.py --ID USMLE_4 --gpu 4 --RA_method Only_RA --dataset USMLE --n_docs 10   >/dev/null 2>&1 &
2024-06-18 17:49:55,992 test: acc 60.25, f1 60.08, precision 60.25, recall 60.32, old_doc_len:0.0, new_doc_len:0.0, hallucination: 0.39 



nohup  python run.py --ID USMLE_3 --gpu 5 --RA_method Gate_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 3 --train_eval 100  >/dev/null 2>&1 &
2024-06-18 14:31:52,449 test: acc 65.83, f1 65.58, precision 65.77, recall 65.74, old_doc_len:0.0, new_doc_len:0.0, hallucination: 0.55 

nohup  python run.py --ID USMLE_2 --gpu 4 --RA_method MI_RA   --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 3 --train_eval 100   >/dev/null 2>&1 &
2024-06-18 02:39:12,299 test: acc 60.25, f1 60.09, precision 60.27, recall 60.35, old_doc_len:2022.56, new_doc_len:2010.06, hallucination: 0.39 


nohup  python run.py --ID USMLE_0 --gpu 4 --RA_method Gate_MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 3 --train_eval 100   >/dev/null 2>&1 &
2024-06-17 20:35:35,726 test: acc 66.06, f1 65.83, precision 66.04, recall 65.98, old_doc_len:2022.56, new_doc_len:2022.56, hallucination: 0.39 

nohup  python run.py --ID USMLE_1 --gpu 5 --RA_method Gate_MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 3 --train_eval 1000  >/dev/null 2>&1 &
2024-06-17 14:33:01,976 test: acc 67.16, f1 66.92, precision 67.11, recall 67.02, old_doc_len:2022.56, new_doc_len:2022.56, hallucination: 0.39 


nohup  python run.py --ID USMLE_0 --gpu 4 --RA_method Gate_MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 3 --train_eval 100  --quantile_num 0.97  >/dev/null 2>&1 &




nohup  python run.py --ID USMLE_0 --gpu 4 --RA_method MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 3 --train_eval 1000   >/dev/null 2>&1 &
 best_step:2000, best_performce: 60.25 

nohup  python run.py --ID USMLE_1 --gpu 5 --RA_method MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1 --train_eval 1000  >/dev/null 2>&1 &
 best_step:2000, best_performce: 60.41 

nohup  python run.py --ID USMLE_2 --gpu 6 --RA_method MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard   --train_eval 1000  >/dev/null 2>&1 &
 best_step:5000, best_performce: 60.02 

nohup  python run.py --ID USMLE_4 --gpu 7 --RA_method MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard   --train_eval 1000  --quantile_num 1 >/dev/null 2>&1 &
 best_step:3000, best_performce: 54.75 



nohup  python run.py --ID USMLE_1 --gpu 5 --RA_method MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 30 --train_eval 100 --quantile_num 0.97  >/dev/null 2>&1 &

nohup  python run.py --ID USMLE_2 --gpu 6 --RA_method MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 3 --train_eval 100 --quantile_num 0.97  >/dev/null 2>&1 &


===================================================================================================================================================================================================
# MedMCQA


nohup  python run.py --ID MedMCQA_1 --gpu 4 --RA_method MI_RA --dataset MedMCQA --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 3 --train_eval 1000  >/dev/null 2>&1 &
 best_step:2000, best_performce: 

nohup  python run.py --ID MedMCQA_2 --gpu 5 --RA_method Gate_MI_RA --dataset MedMCQA --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 3 --train_eval 1000  >/dev/null 2>&1 &
best_step:2000, best_performce: 





===================================================================================================================================================================================================
# HEADQA


nohup  python run.py --ID HEADQA_1  --gpu 6 --RA_method MI_RA --dataset HEADQA  --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 3 --train_eval 1000  >/dev/null 2>&1 &
 best_step:2000, best_performce: 44.97 

nohup  python run.py --ID HEADQA_2  --gpu 7 --RA_method Gate_MI_RA --dataset HEADQA  --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 3 --train_eval 1000  >/dev/null 2>&1 &
 best_step:2000, best_performce: 63.89 

nohup  python run.py --ID HEADQA_3  --gpu 7 --RA_method MI_RA --dataset HEADQA  --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 3 --train_eval 1000  >/dev/null 2>&1 &

