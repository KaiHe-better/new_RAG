# # llama2-7b_USMLE
# nohup python run.py --ID USMLE_00  --gpu 4 --config llama2-7b_USMLE.yaml --dataset USMLE  >/dev/null 2>&1 &  
# 2024-05-04 13:40:54,956 test: acc 31.97, f1 25.77, precision 30.72, recall 34.55, old_doc_len:0.0, new_doc_len:0.0, hallucination: 0.0 



# # llama2-7b_USMLE_RA
# nohup python run.py --ID USMLE_0  --gpu 7 --config llama2-7b_USMLE_RA.yaml --dataset USMLE  >/dev/null 2>&1 &  
# 2024-05-04 10:48:32,949 test: acc 38.18, f1 38.1, precision 38.18, recall 38.58, old_doc_len:1253.7511773940346, new_doc_len:0.0, hallucination: 0.0 

# nohup python run.py --ID USMLE_0  --gpu 7 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --n_docs 10 >/dev/null 2>&1 &  
# 2024-05-06 11:23:41,434 test: acc 35.9, f1 35.82, precision 35.91, recall 36.86, old_doc_len:2481.970957613815, new_doc_len:0.0, hallucination: 1.1 



# # llama2-7b_USMLE_MI_RA  --n_docs 5
# nohup python run.py --ID USMLE_06_     --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE  >/dev/null 2>&1 &  
# test: acc 37.86, f1 37.78, precision 37.87, recall 38.21, old_doc_len:1253.7511773940346, new_doc_len:827.265306122449

# nohup python run.py --ID USMLE_07_Loss --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 500 >/dev/null 2>&1 &
# 问题:保留了所有的检索结果
# 2024-05-04 09:42:01,634 test: acc 38.18, f1 38.1, precision 38.18, recall 38.58, old_doc_len:1253.7511773940346, new_doc_len:1253.256671899529, hallucination: 0.0 

# nohup python run.py --ID USMLE_0 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+len_penalty --len_penalty_weight 100 >/dev/null 2>&1 &
# 问题:保留了所有的检索结果
# 2024-05-05 01:34:19,214 test: acc 38.41, f1 38.31, precision 38.34, recall 38.8, old_doc_len:1253.7511773940346, new_doc_len:1214.9207221350077, hallucination: 0.0 

# nohup python run.py --ID USMLE_1 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_hard+len_penalty --len_penalty_weight 100 >/dev/null 2>&1 &
# 问题:保留了所有的检索结果
# 2024-05-05 01:44:45,764 test: acc 38.26, f1 38.16, precision 38.23, recall 38.63, old_doc_len:1253.7511773940346, new_doc_len:1227.541601255887, hallucination: 0.0 

# nohup python run.py --ID USMLE_2 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1     --len_penalty_weight 100 >/dev/null 2>&1 &
# 问题:保留了所有的检索结果
# 2024-05-05 06:38:47,023 test: acc 38.41, f1 38.33, precision 38.42, recall 38.8, old_doc_len:1253.7511773940346, new_doc_len:1247.513343799058, hallucination: 0.0 

# nohup python run.py --ID USMLE_2 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+kl_hard+len_penalty --soft_weight 0.7 --hard_weight 0.3 --len_penalty_weight 100 >/dev/null 2>&1 &
# 问题:保留了所有的检索结果
# 2024-05-05 05:17:11,951 test: acc 38.33, f1 38.26, precision 38.36, recall 38.74, old_doc_len:1253.7511773940346, new_doc_len:1240.298273155416, hallucination: 0.0 

# nohup python run.py --ID USMLE_0 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+len_penalty --len_penalty_weight 10 >/dev/null 2>&1 &
# 问题:保留的太多了
# 2024-05-06 01:00:12,807 test: acc 38.33, f1 38.23, precision 38.25, recall 38.7, old_doc_len:1253.7511773940346, new_doc_len:1123.4356357927786, hallucination: 0.0 

# nohup python run.py --ID USMLE_1 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_hard+len_penalty --len_penalty_weight 10 >/dev/null 2>&1 &
# 问题:保留的太多了
# 2024-05-06 04:03:40,874 test: acc 38.33, f1 38.26, precision 38.33, recall 38.73, old_doc_len:1253.7511773940346, new_doc_len:1235.861852433281, hallucination: 0.0 

# nohup python run.py --ID USMLE_2 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1     --len_penalty_weight 10 >/dev/null 2>&1 &
# 问题:保留的太多了
# 2024-05-06 04:00:10,466 test: acc 38.57, f1 38.52, precision 38.64, recall 39.07, old_doc_len:1253.7511773940346, new_doc_len:1193.9246467817895, hallucination: 0.0 

# nohup python run.py --ID USMLE_3 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --loss_list kl_soft+kl_hard+len_penalty --soft_weight 0.7 --hard_weight 0.3 --len_penalty_weight 10 >/dev/null 2>&1 &
# 问题:保留的太多了
# 2024-05-06 04:47:22,311 test: acc 38.33, f1 38.25, precision 38.29, recall 38.67, old_doc_len:1253.7511773940346, new_doc_len:1191.190737833595, hallucination: 0.0 



# # llama2-7b_USMLE_MI_RA  --n_docs 10

# nohup python run.py --ID USMLE_0 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+len_penalty --len_penalty_weight 5 >/dev/null 2>&1 &
# 2024-05-06 15:51:21,175 test: acc 37.63, f1 37.57, precision 37.87, recall 38.57, old_doc_len:2481.970957613815, new_doc_len:1878.870486656201, hallucination: 0.0 

# nohup python run.py --ID USMLE_1 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_hard+len_penalty --len_penalty_weight 5 >/dev/null 2>&1 &
# 2024-05-06 12:04:36,145 test: acc 36.84, f1 36.78, precision 37.02, recall 37.68, old_doc_len:2481.970957613815, new_doc_len:1896.5, hallucination: 0.08 

# nohup python run.py --ID USMLE_2 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1     --len_penalty_weight 5 >/dev/null 2>&1 &
# 2024-05-07 10:06:44,816 test: acc 37.47, f1 37.42, precision 37.7, recall 38.49, old_doc_len:2481.970957613815, new_doc_len:2155.784929356358, hallucination: 0.08 

# nohup python run.py --ID USMLE_3 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 0.7 --hard_weight 0.3 --len_penalty_weight 5 >/dev/null 2>&1 &
# 2024-05-06 15:32:48,852 test: acc 37.16, f1 37.09, precision 37.35, recall 37.9, old_doc_len:2481.970957613815, new_doc_len:1786.539246467818, hallucination: 0.08 




# # new llama2-7b_USMLE_MI_RA  --n_docs 10

# nohup python run.py --ID USMLE_0 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --gate_weight 1  --len_penalty_weight 4 >/dev/null 2>&1 &
# 2024-05-07 23:29:31,590 test: acc 49.49, f1 47.4, precision 48.63, recall 48.87, old_doc_len:2480.0533751962325, new_doc_len:872.234693877551, hallucination: 0.08 
# 2024-05-07 23:29:31,590 cost_time: 29.669101238250732 , gate_res_list: 0.648075412411626, 825 / 1273 

# nohup python run.py --ID USMLE_3 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --gate_weight 2  --len_penalty_weight 4 >/dev/null 2>&1 &
# 2024-05-08 03:41:56,168 test: acc 49.49, f1 47.37, precision 48.63, recall 48.82, old_doc_len:2480.0533751962325, new_doc_len:892.6883830455259, hallucination: 0.08 
# 2024-05-08 03:41:56,168 cost_time: 50.711981503168744 , gate_res_list: 0.6543597800471328, 833 / 1273 

# nohup python run.py --ID USMLE_0 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard             --soft_weight 1 --hard_weight 1  >/dev/null 2>&1 &
# 2024-05-09 08:05:25,223 test: acc 49.57, f1 47.46, precision 48.66, recall 48.36, old_doc_len:2480.0533751962325, new_doc_len:808.4748822605966, hallucination: 0.0 
# 2024-05-09 08:05:25,223 cost_time: 29.41408345301946 , gate_res_list: 0.6575019638648861, 837 / 1273 

# nohup python run.py --ID USMLE_1 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --len_penalty_weight 1 >/dev/null 2>&1 &
# 2024-05-10 03:37:33,006 test: acc 49.33, f1 47.21, precision 48.49, recall 48.41, old_doc_len:2480.0533751962325, new_doc_len:951.1287284144427, hallucination: 0.0 
# 2024-05-10 03:37:33,006 cost_time: 44.86397190888723 , gate_res_list: 0.6622152395915161, 843 / 1273 

# nohup python run.py --ID USMLE_2 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --len_penalty_weight 2 >/dev/null 2>&1 &
# 2024-05-08 19:45:30,506 test: acc 49.49, f1 47.37, precision 48.57, recall 48.78, old_doc_len:2480.0533751962325, new_doc_len:846.4073783359497, hallucination: 0.0 
# 2024-05-08 19:45:30,506 cost_time: 32.293471018473305 , gate_res_list: 0.6567164179104478, 836 / 1273 

# nohup python run.py --ID USMLE_3 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --len_penalty_weight 3 >/dev/null 2>&1 &
# 2024-05-08 20:41:18,559 test: acc 49.65, f1 47.55, precision 48.72, recall 48.99, old_doc_len:2480.0533751962325, new_doc_len:864.5368916797488, hallucination: 0.08 
# 2024-05-08 20:41:18,559 cost_time: 35.15572950442632 , gate_res_list: 0.6567164179104478, 836 / 1273 



# # new llama2-7b_USMLE_MI_RA  --n_docs 5

# nohup python run.py --ID USMLE_0 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 5 --loss_list kl_soft+kl_hard             --soft_weight 1 --hard_weight 1  >/dev/null 2>&1 &
# 2024-05-11 14:14:03,076 test: acc 50.27, f1 48.28, precision 49.24, recall 48.98, old_doc_len:1252.7605965463108, new_doc_len:542.4568288854003, hallucination: 0.0 
# 2024-05-11 14:14:03,077 cost_time: 26.357947031656902 , gate_res_list: 0.6614296936370778, 842 / 1273 

# nohup python run.py --ID USMLE_1 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 5 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --len_penalty_weight 5 >/dev/null 2>&1 &
# 2024-05-10 23:00:17,128 test: acc 50.2, f1 48.16, precision 49.14, recall 48.92, old_doc_len:1252.7605965463108, new_doc_len:536.6758241758242, hallucination: 0.0 
# 2024-05-10 23:00:17,128 cost_time: 29.71108977397283 , gate_res_list: 0.6575019638648861, 837 / 1273 

# nohup python run.py --ID USMLE_2 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 5 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --len_penalty_weight 8 >/dev/null 2>&1 &
# 2024-05-10 21:50:18,816 test: acc 50.2, f1 48.16, precision 49.14, recall 48.92, old_doc_len:1252.7605965463108, new_doc_len:536.6758241758242, hallucination: 0.0 
# 2024-05-10 21:50:18,816 cost_time: 26.213221351305645 , gate_res_list: 0.6575019638648861, 837 / 1273 

# nohup python run.py --ID USMLE_3 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 5 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --len_penalty_weight 10 >/dev/null 2>&1 &
# 2024-05-10 22:30:25,900 test: acc 50.2, f1 48.16, precision 49.14, recall 48.92, old_doc_len:1252.7605965463108, new_doc_len:536.6758241758242, hallucination: 0.0 
# 2024-05-10 22:30:25,900 cost_time: 29.351856935024262 , gate_res_list: 0.6575019638648861, 837 / 1273 


# # new llama3-8b_USMLE_MI_RA 


# nohup python run.py --ID USMLE_0 --gpu 4 --config llama3-8b_USMLE_RA.yaml --dataset USMLE --n_docs 5   >/dev/null 2>&1 &
# 2024-05-12 19:57:45,893 test: acc 53.73, f1 53.62, precision 53.79, recall 54.13, old_doc_len:1021.0675039246468, new_doc_len:1021.0675039246468, hallucination: 10.68 

# nohup python run.py --ID USMLE_01 --gpu 7 --config llama3-8b_USMLE_RA.yaml --dataset USMLE --n_docs 10  >/dev/null 2>&1 &
# 2024-05-12 20:01:40,753 test: acc 59.86, f1 59.69, precision 59.88, recall 59.95, old_doc_len:2020.9819466248039, new_doc_len:2020.9819466248039, hallucination: 0.31 



# nohup python run.py --ID USMLE_2 --gpu 7 --config llama3-8b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 5  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 10 >/dev/null 2>&1 &
# 2024-05-13 01:25:04,476 test: acc 64.89, f1 64.67, precision 64.93, recall 64.97, old_doc_len:1021.0675039246468, new_doc_len:190.43720565149135, hallucination: 5.97 

# nohup python run.py --ID USMLE_3 --gpu 4 --config llama3-8b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 10 >/dev/null 2>&1 &
# 2024-05-15 09:42:19,157 test: acc 66.93, f1 66.69, precision 66.9, recall 66.79, old_doc_len:2020.9819466248039, new_doc_len:381.6773940345369, hallucination: 0.39 


# # 这两个训练时间太长，这2个貌似没区别， 000比001稍好
# nohup python run.py --ID USMLE_000 --gpu 4 --config llama3-8b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 3 >/dev/null 2>&1 &
# 2024-05-23 08:18:46,111 test: acc 67.48, f1 67.26, precision 67.48, recall 67.33, old_doc_len:2020.9819466248039, new_doc_len:290.8006279434851, hallucination: 1.73 
# 67.01

# nohup python run.py --ID USMLE_001 --gpu 5 --config llama3-8b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard >/dev/null 2>&1 &
# 2024-05-23 04:23:07,022 test: acc 67.48, f1 67.25, precision 67.45, recall 67.32, old_doc_len:2020.9819466248039, new_doc_len:347.8689167974882, hallucination: 0.39 
# 67.01


===================================================================================================================================================================================================
# USMLE
nohup  python run.py --ID USMLE_5 --gpu 6 --RA_method No_RA   --dataset USMLE --n_docs 10    >/dev/null 2>&1 &
2024-06-17 22:57:06,374 test: acc 58.99, f1 58.79, precision 59.05, recall 59.42, old_doc_len:0.0, new_doc_len:0.0, hallucination: 2.44 

nohup  python run.py --ID USMLE_4 --gpu 4 --RA_method Only_RA --dataset USMLE --n_docs 10   >/dev/null 2>&1 &
2024-06-18 17:49:55,992 test: acc 60.25, f1 60.08, precision 60.25, recall 60.32, old_doc_len:0.0, new_doc_len:0.0, hallucination: 0.39 

nohup  python run.py --ID USMLE_11 --gpu 5 --RA_method Gate_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard  --quantile_num 0.95 --train_batch_size 8 --test_batch_size 8  >/dev/null 2>&1 &
2024-06-24 00:31:01,626 test: acc 66.38, f1 66.14, precision 66.37, recall 66.3, old_doc_len:0.0, new_doc_len:0.0, hallucination: 0.39 
training process num: 12407/1273,  best_step:3500

nohup  python run.py --ID USMLE_1 --gpu 5 --RA_method Gate_MI_RA --dataset USMLE --quantile_num 0.99 --if_hierarchical_retrieval True --train_batch_size 8 --test_batch_size 8  >/dev/null 2>&1 &
2024-06-26 11:57:36,064 test: acc 67.09, f1 66.88, precision 67.14, recall 66.99, old_doc_len:2022.52, new_doc_len:1197.32, hallucination: 1.65 
 best_step:32000



# nohup  python run.py --ID USMLE_0 --gpu 4 --RA_method MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 3 --train_eval 1000   >/dev/null 2>&1 &
#  best_step:2000, best_performce: 60.25 

# nohup  python run.py --ID USMLE_0 --gpu 4 --RA_method Gate_MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 3 --train_eval 100  --quantile_num 1  >/dev/null 2>&1 &
# 2024-06-22 10:40:49,693 test: acc 64.57, f1 64.38, precision 64.71, recall 64.76, old_doc_len:2022.56, new_doc_len:1001.57, hallucination: 4.79 

# nohup  python run.py --ID USMLE_0 --gpu 4 --RA_method Gate_MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 300 --quantile_num 1.0  --train_batch_size 8 --test_batch_size 8  >/dev/null 2>&1 &
# 2024-06-23 13:14:22,833 test: acc 64.81, f1 64.63, precision 64.95, recall 64.93, old_doc_len:2022.52, new_doc_len:1123.01, hallucination: 4.24 

# nohup  python run.py --ID USMLE_1 --gpu 5 --RA_method Gate_MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 300 --quantile_num 0.97 --train_batch_size 8 --test_batch_size 8  >/dev/null 2>&1 &
# 2024-06-23 13:43:57,231 test: acc 66.46, f1 66.21, precision 66.45, recall 66.34, old_doc_len:2022.52, new_doc_len:2022.52, hallucination: 0.55 

# nohup  python run.py --ID USMLE_4 --gpu 6 --RA_method Gate_MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard  --quantile_num 0.97 --train_batch_size 8 --test_batch_size 8  >/dev/null 2>&1 &
# 2024-06-23 05:22:29,060 test: acc 66.77, f1 66.57, precision 66.83, recall 66.75, old_doc_len:2022.52, new_doc_len:1888.29, hallucination: 0.71 

# nohup  python run.py --ID USMLE_5 --gpu 7 --RA_method Gate_MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard  --quantile_num 0.95 --train_batch_size 8 --test_batch_size 8  >/dev/null 2>&1 &
# 2024-06-23 05:29:02,677 test: acc 66.61, f1 66.39, precision 66.64, recall 66.54, old_doc_len:2022.52, new_doc_len:1997.19, hallucination: 0.47 



# # quantile_num 越小取的越多
# nohup  python run.py --ID USMLE_6 --gpu 4 --RA_method Gate_MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard  --quantile_num 0.99 --train_batch_size 8 --test_batch_size 8  >/dev/null 2>&1 &
# 2024-06-24 10:49:57,488 test: acc 67.09, f1 66.89, precision 67.13, recall 67.0, old_doc_len:2022.52, new_doc_len:1683.44, hallucination: 1.57 
#  best_step:12000

# nohup  python run.py --ID USMLE_7 --gpu 7 --RA_method Gate_MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard  --quantile_num 0.98 --train_batch_size 8 --test_batch_size 8  >/dev/null 2>&1 &
# 2024-06-24 07:54:06,794 test: acc 66.54, f1 66.31, precision 66.54, recall 66.41, old_doc_len:2022.52, new_doc_len:1999.42, hallucination: 0.63 
#  best_step:9500

# nohup  python run.py --ID USMLE_8 --gpu 6 --RA_method Gate_MI_RA --dataset USMLE --n_docs 10  --loss_list kl_soft+kl_hard  --quantile_num 0.97 --train_batch_size 8 --test_batch_size 8  >/dev/null 2>&1 &
# 2024-06-24 00:00:40,889 test: acc 66.77, f1 66.57, precision 66.83, recall 66.75, old_doc_len:2022.52, new_doc_len:1886.91, hallucination: 0.71 
# best_step:8000



# 关掉了train 和 eval
nohup  python run.py --ID USMLE_0 --gpu 4 --RA_method Gate_MI_RA --dataset USMLE --quantile_num 1 --if_hierarchical_retrieval True --train_batch_size 8 --test_batch_size 8   >/dev/null 2>&1 &
2024-06-26 15:37:10,720 test: acc 66.77, f1 66.51, precision 66.75, recall 66.7, old_doc_len:2022.52, new_doc_len:1139.97, hallucination: 2.12 
 best_step:32000

nohup  python run.py --ID USMLE_1 --gpu 5 --RA_method Gate_MI_RA --dataset USMLE --quantile_num 0.99 --if_hierarchical_retrieval True --train_batch_size 8 --test_batch_size 8  >/dev/null 2>&1 &
2024-06-26 11:57:36,064 test: acc 67.09, f1 66.88, precision 67.14, recall 66.99, old_doc_len:2022.52, new_doc_len:1197.32, hallucination: 1.65 
 best_step:32000

# # --if_hierarchical_retrieval False
# nohup  python run.py --ID USMLE_2 --gpu 6 --RA_method Gate_MI_RA --dataset USMLE --quantile_num 1 --train_batch_size 8 --test_batch_size 8  >/dev/null 2>&1 &
# 2024-06-25 10:06:39,682 test: acc 65.36, f1 65.16, precision 65.5, recall 65.45, old_doc_len:2022.52, new_doc_len:1065.41, hallucination: 5.81 
#  best_step:15500

# nohup  python run.py --ID USMLE_3 --gpu 7 --RA_method Gate_MI_RA --dataset USMLE --quantile_num 0.99 --train_batch_size 8 --test_batch_size 8  >/dev/null 2>&1 &
# 2024-06-25 10:07:18,063 test: acc 65.75, f1 65.56, precision 65.9, recall 65.82, old_doc_len:2022.52, new_doc_len:1120.74, hallucination: 5.26 
#  best_step:15500






===================================================================================================================================================================================================
# MedMCQA


nohup  python run.py --ID MedMCQA_1 --gpu 7   --RA_method No_RA      --dataset MedMCQA   >/dev/null 2>&1 &
2024-06-27 13:30:28,239 test: acc 47.57, f1 45.69, precision 45.59, recall 50.54, old_doc_len:0.0, new_doc_len:0.0, hallucination: 4.85 

nohup  python run.py --ID MedMCQA_2 --gpu 7   --RA_method Only_RA    --dataset MedMCQA   >/dev/null 2>&1 &
2024-06-27 13:52:42,412 test: acc 43.56, f1 36.04, precision 37.9, recall 64.08, old_doc_len:0.0, new_doc_len:0.0, hallucination: 0.5 

nohup  python run.py --ID MedMCQA_9 --gpu 7   --RA_method MI_RA      --dataset MedMCQA  --quantile_num 0.80 >/dev/null 2>&1 &
2024-06-27 22:00:38,782 test: acc 44.75, f1 38.48, precision 39.44, recall 62.91, old_doc_len:1533.97, new_doc_len:1494.72, hallucination: 0.14 

nohup  python run.py --ID MedMCQA_4 --gpu 5   --RA_method Gate_RA    --dataset MedMCQA   >/dev/null 2>&1 &
2024-06-28 07:13:06,284 test: acc 61.42, f1 58.97, precision 57.01, recall 75.6, old_doc_len:0.0, new_doc_len:0.0, hallucination: 0.43 
 best_step:24500

nohup  python run.py --ID MedMCQA_5_2 --gpu 4  --RA_method Gate_MI_RA --dataset MedMCQA --quantile_num 0.80  >/dev/null 2>&1 &
2024-06-29 15:32:15,900 test: acc 61.68, f1 59.39, precision 57.41, recall 75.12, old_doc_len:1533.97, new_doc_len:1439.13, hallucination: 0.12 
 best_step:24000




# nohup  python run.py --ID MedMCQA_5_2 --gpu 5  --RA_method Gate_MI_RA --dataset MedMCQA --quantile_num 0.825  >/dev/null 2>&1 &
# 2024-06-29 04:36:22,559 test: acc 61.34, f1 58.91, precision 56.93, recall 76.04, old_doc_len:1533.97, new_doc_len:1357.5, hallucination: 0.17 
#  best_step:17500

# nohup  python run.py --ID MedMCQA_5_2 --gpu 6  --RA_method Gate_MI_RA --dataset MedMCQA --quantile_num 0.85  >/dev/null 2>&1 &
# 2024-06-29 10:49:09,792 test: acc 61.37, f1 59.01, precision 56.99, recall 76.28, old_doc_len:1533.97, new_doc_len:1331.3, hallucination: 0.22 
#  best_step:24000





===================================================================================================================================================================================================
# HEADQA

nohup  python run.py --ID HEADQA_0  --gpu 4 --RA_method No_RA --dataset HEADQA   >/dev/null 2>&1 &
2024-06-26 19:06:58,740 test: acc 55.8, f1 55.17, precision 55.47, recall 58.0, old_doc_len:0.0, new_doc_len:0.0, hallucination: 11.01 

nohup  python run.py --ID HEADQA_0  --gpu 5 --RA_method Only_RA --dataset HEADQA   >/dev/null 2>&1 &
2024-06-26 19:21:06,086 test: acc 45.44, f1 44.01, precision 45.69, recall 67.86, old_doc_len:0.0, new_doc_len:0.0, hallucination: 3.14 
10m

nohup  python run.py --ID HEADQA_3  --gpu 6 --RA_method MI_RA --dataset HEADQA   >/dev/null 2>&1 &
2024-06-26 19:57:12,748 test: acc 38.73, f1 35.08, precision 39.14, recall 62.71, old_doc_len:1567.07, new_doc_len:941.66, hallucination: 4.45 
 best_step:500

nohup  python run.py --ID HEADQA_1  --gpu 4 --RA_method Gate_RA --dataset HEADQA   >/dev/null 2>&1 &
2024-06-27 08:13:36,185 test: acc 64.73, f1 65.21, precision 64.67, recall 78.16, old_doc_len:0.0, new_doc_len:0.0, hallucination: 1.5 
 best_step:22000, 13m

nohup  python run.py --ID HEADQA_2  --gpu 6 --RA_method Gate_MI_RA --dataset HEADQA --quantile_num 0.80  >/dev/null 2>&1 &
2024-06-30 10:25:19,751 test: acc 65.06, f1 65.59, precision 65.02, recall 77.9, old_doc_len:1567.07, new_doc_len:1405.55, hallucination: 0.4                                   
 best_step:22500


===================================================================================================================================================================================================
# PopQA

nohup  python run.py --ID PopQA_0  --gpu 4 --RA_method No_RA   --dataset PopQA   >/dev/null 2>&1 &
2024-07-03 11:51:08,027 test: f1 28.15, EM : 22.92, old_doc_len:0.0, new_doc_len:0.0
# 2024-07-13 12:07:19,941 test: f1 28.33, EM : 27.89, old_doc_len:0.0, new_doc_len:0.0

nohup  python run.py --ID PopQA_1  --gpu 5 --RA_method Only_RA --dataset PopQA   >/dev/null 2>&1 &
2024-07-03 15:43:56,184 test: f1 36.09, EM : 26.77, old_doc_len:0.0, new_doc_len:0.0
# 2024-07-14 15:30:49,685 test: f1 36.35, EM : 52.56, old_doc_len:0.0, new_doc_len:0.0


nohup  python run.py --ID PopQA_2_6_2  --gpu 4 --RA_method MI_RA --dataset PopQA --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
2024-07-09 19:50:43,301 test: f1 37.72, EM : 29.64, old_doc_len:1126.01, new_doc_len:813.65
#  best_step:3000

nohup  python run.py --ID PopQA_3  --gpu 6 --RA_method Gate_RA --dataset PopQA --gate_weight_0 4 --gate_weight_1 1 --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
2024-07-10 15:19:22,036 test: f1 35.43, EM : 27.68, old_doc_len:0.0, new_doc_len:0.0
#  best_step:3000

# GCP
nohup  python run.py --ID PopQA_4_5  --gpu 6 --RA_method Gate_MI_RA --dataset PopQA --gate_weight_0 3 --gate_weight_1 1 --quantile_num 0.5  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1    >/dev/null 2>&1 &
2024-07-11 08:43:16,052 test: f1 39.78, EM : 31.32, old_doc_len:1126.01, new_doc_len:1171.47
#  best_step:500


nohup  python run.py --ID PopQA_2  --gpu 4 --RA_method MI_RA --dataset PopQA --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
nohup  python run.py --ID PopQA_3  --gpu 6 --RA_method Gate_RA --dataset PopQA --gate_weight_0 4 --gate_weight_1 1 --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
nohup  python run.py --ID PopQA_4  --gpu 6 --RA_method Gate_MI_RA --dataset PopQA --gate_weight_0 3 --gate_weight_1 1 --quantile_num 0.5  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1    >/dev/null 2>&1 &


===================================================================================================================================================================================================

nohup  python run.py --ID WebQA_1_1  --gpu 4 --RA_method No_RA   --dataset WebQA   >/dev/null 2>&1 &
2024-07-11 15:29:54,987 test: f1 27.83, EM : 46.01, old_doc_len:0.0, new_doc_len:0.0

nohup  python run.py --ID WebQA_2  --gpu 4 --RA_method Only_RA --dataset WebQA   >/dev/null 2>&1 &
2024-07-11 15:47:27,230 test: f1 21.74, EM : 37.7, old_doc_len:0.0, new_doc_len:0.0


# nohup  python run.py --ID WebQA_3    --gpu 6 --RA_method MI_RA    --dataset WebQA --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
# 2024-07-11 16:32:43,960 test: f1 23.79, EM : 36.86, old_doc_len:1293.44, new_doc_len:1277.6
#  best_step:500

# nohup  python run.py --ID WebQA_3_1  --gpu 7 --RA_method MI_RA    --dataset WebQA --quantile_num 0.7  --loss_list kl_soft+kl_hard  >/dev/null 2>&1 &
# 2024-07-11 16:57:24,517 test: f1 22.31, EM : 34.94, old_doc_len:1293.44, new_doc_len:1047.62
#  best_step:500

# nohup  python run.py --ID WebQA_3_2  --gpu 4 --RA_method MI_RA    --dataset WebQA --quantile_num 0.85  --loss_list kl_soft+kl_hard  >/dev/null 2>&1 &
# 2024-07-11 16:27:45,277 test: f1 23.81, EM : 36.52, old_doc_len:1293.44, new_doc_len:1307.22
#  best_step:500

# nohup  python run.py --ID WebQA_3_3    --gpu 4 --RA_method MI_RA    --dataset WebQA --quantile_num 0.6  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
# 2024-07-11 21:27:52,419 test: f1 23.85, EM : 36.66, old_doc_len:1293.44, new_doc_len:1352.3
#  best_step:500


nohup  python run.py --ID WebQA_3_3    --gpu 6 --RA_method Gate_MI_RA    --dataset WebQA --quantile_num 0.6  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
2024-07-11 21:43:31,309 test: f1 24.03, EM : 37.01, old_doc_len:1293.44, new_doc_len:1346.06
 best_step:500

nohup  python run.py --ID WebQA_3_3    --gpu 7 --RA_method Gate_MI_RA    --dataset WebQA --quantile_num 0.6  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
2024-07-11 21:43:07,730 test: f1 24.08, EM : 37.11, old_doc_len:1293.44, new_doc_len:1346.14
 best_step:500

~~~~~~~~~
# serious 

nohup  python run.py --ID WebQA_0  --gpu 6 --RA_method No_RA   --dataset WebQA   >/dev/null 2>&1 &
nohup  python run.py --ID WebQA_1  --gpu 7 --RA_method Only_RA --dataset WebQA   >/dev/null 2>&1 &

nohup  python run.py --ID WebQA_2  --gpu 4 --RA_method MI_RA      --dataset WebQA --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
nohup  python run.py --ID WebQA_3  --gpu 6 --RA_method Gate_RA    --dataset WebQA --gate_weight_0 3 --gate_weight_1 1 --quantile_num 0.8  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
nohup  python run.py --ID WebQA_4  --gpu 7 --RA_method Gate_MI_RA --dataset WebQA --gate_weight_0 3 --gate_weight_1 1 --quantile_num 0.85  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &




===================================================================================================================================================================================================

nohup  python run.py --ID TriviaQA_0  --gpu 4 --RA_method No_RA   --dataset TriviaQA   >/dev/null 2>&1 &
2024-07-12 15:14:08,173 test: f1 51.73, EM : 59.29, old_doc_len:0.0, new_doc_len:0.0

nohup  python run.py --ID TriviaQA_1  --gpu 4 --RA_method Only_RA --dataset TriviaQA   >/dev/null 2>&1 &
2024-07-12 21:42:38,524 test: f1 50.28, EM : 59.52, old_doc_len:0.0, new_doc_len:0.0

nohup  python run.py --ID TriviaQA_2  --gpu 4 --RA_method MI_RA   --dataset TriviaQA --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
2024-07-15 00:30:24,070 test: f1 51.45, EM : 58.38, old_doc_len:1389.89, new_doc_len:1404.65
 best_step:500

nohup  python run.py --ID TriviaQA_3  --gpu 6 --RA_method Gate_RA   --dataset TriviaQA  --gate_weight_0 1 --gate_weight_1 1   >/dev/null 2>&1 &
2024-07-15 01:34:31,458 test: f1 52.16, EM : 60.95, old_doc_len:0.0, new_doc_len:0.0
 best_step:2000

nohup  python run.py --ID TriviaQA_4_1  --gpu 6 --RA_method Gate_MI_RA  --dataset TriviaQA --gate_weight_0 1 --gate_weight_1 1.2 --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
MI_60.87

~~~~~

nohup  python run.py --ID TriviaQA_0  --gpu 4 --RA_method No_RA   --dataset TriviaQA   >/dev/null 2>&1 &
2024-07-16 23:10:20,132 test: f1 51.73, EM : 41.69, old_doc_len:0.0, new_doc_len:0.0

nohup  python run.py --ID TriviaQA_1  --gpu 4 --RA_method Only_RA --dataset TriviaQA   >/dev/null 2>&1 &
2024-07-17 00:09:34,195 test: f1 50.28, EM : 39.98, old_doc_len:0.0, new_doc_len:0.0


nohup  python run.py --ID TriviaQA_2   --gpu 4 --RA_method MI_RA      --dataset TriviaQA --quantile_num 0.7 --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
MI_41.03
nohup  python run.py --ID TriviaQA_3   --gpu 6 --RA_method Gate_RA    --dataset TriviaQA --gate_weight_0 1  --gate_weight_1 1   >/dev/null 2>&1 &
MI_41.08
nohup  python run.py --ID TriviaQA_4_1 --gpu 7 --RA_method Gate_MI_RA --dataset TriviaQA --gate_weight_0 1  --gate_weight_1 1.2 --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
MI_40.79


===================================================================================================================================================================================================
# GCP
nohup  python run.py --ID NQ_0  --gpu 4 --RA_method No_RA   --dataset NQ   >/dev/null 2>&1 &
2024-07-12 20:30:42,692 test: f1 24.62, EM : 33.24, old_doc_len:0.0, new_doc_len:0.0

nohup  python run.py --ID NQ_1  --gpu 6 --RA_method Only_RA --dataset NQ   >/dev/null 2>&1 &
2024-07-12 20:50:17,985 test: f1 28.57, EM : 40.53, old_doc_len:0.0, new_doc_len:0.0

nohup  python run.py --ID NQ_2  --gpu 4 --RA_method MI_RA   --dataset NQ --quantile_num 0.6  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
2024-07-15 07:12:17,300 test: f1 29.49, EM : 39.67, old_doc_len:1359.77, new_doc_len:1412.27

nohup  python run.py --ID NQ_2  --gpu 6 --RA_method Gate_RA   --dataset NQ --gate_weight_0 2 --gate_weight_1 1  >/dev/null 2>&1 &
2024-07-15 05:50:39,103 test: f1 28.45, EM : 40.72, old_doc_len:0.0, new_doc_len:0.0
gate_res_list: 1.0, 3610 / 3610 

nohup  python run.py --ID NQ_4  --gpu 7 --RA_method Gate_MI_RA    --dataset NQ --gate_weight_0 2 --gate_weight_1 1 --quantile_num 0.6  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
2024-07-15 08:19:35,322 test: f1 29.68, EM : 39.67, old_doc_len:1359.77, new_doc_len:1412.05
gate_res_list: 1.0, 3610 / 3610 

# nohup  python run.py --ID NQ_2  --gpu 4 --RA_method MI_RA   --dataset NQ --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
# 2024-07-12 17:26:59,778 test: f1 28.68, EM : 39.39, old_doc_len:1359.77, new_doc_len:1277.6

# nohup  python run.py --ID NQ_2  --gpu 6 --RA_method Gate_RA   --dataset NQ --gate_weight_0 3 --gate_weight_1 1  >/dev/null 2>&1 &
# 2024-07-12 12:14:50,179 test: f1 27.33, EM : 39.03, old_doc_len:0.0, new_doc_len:0.0

# nohup  python run.py --ID NQ_4  --gpu 7 --RA_method Gate_MI_RA    --dataset NQ --gate_weight_0 3 --gate_weight_1 1 --quantile_num 0.85  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
# 2024-07-12 20:51:00,289 test: f1 26.92, EM : 36.81, old_doc_len:1359.77, new_doc_len:983.63,  gate_res_list: 0.32

# nohup  python run.py --ID NQ_2  --gpu 4 --RA_method MI_RA   --dataset NQ --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
# 2024-07-13 14:52:30,287 test: f1 28.8, EM : 39.56, old_doc_len:1359.77, new_doc_len:1279.2
#  best_step:3500

# nohup  python run.py --ID NQ_2  --gpu 6 --RA_method Gate_RA   --dataset NQ --gate_weight_0 1 --gate_weight_1 1  >/dev/null 2>&1 &
# 2024-07-13 05:50:16,656 test: f1 28.45, EM : 40.72, old_doc_len:0.0, new_doc_len:0.0
#  best_step:500, gate_res_list: 1.0

# nohup  python run.py --ID NQ_4  --gpu 7 --RA_method Gate_MI_RA    --dataset NQ --gate_weight_0 1 --gate_weight_1 1 --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
# 2024-07-13 23:00:31,551 test: f1 28.84, EM : 39.45, old_doc_len:1359.77, new_doc_len:1267.75
#  best_step:4500

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nohup  python run.py --ID NQ_0  --gpu 2 --RA_method No_RA   --dataset NQ   >/dev/null 2>&1 &
2024-07-16 22:58:34,884 test: f1 24.62, EM : 14.43, old_doc_len:0.0, new_doc_len:0.0

nohup  python run.py --ID NQ_1  --gpu 3 --RA_method Only_RA --dataset NQ   >/dev/null 2>&1 &
2024-07-16 23:17:55,503 test: f1 28.57, EM : 17.7, old_doc_len:0.0, new_doc_len:0.0


nohup  python run.py --ID NQ_2  --gpu 4 --RA_method MI_RA   --dataset NQ --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
nohup  python run.py --ID NQ_2  --gpu 6 --RA_method Gate_RA   --dataset NQ --gate_weight_0 2.5 --gate_weight_1 1  >/dev/null 2>&1 &
nohup  python run.py --ID NQ_4  --gpu 7 --RA_method Gate_MI_RA    --dataset NQ --gate_weight_0 2 --gate_weight_1 1 --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &


===================================================================================================================================================================================================
# hotpot 

nohup  python run.py --ID Hotpot_0  --gpu 2 --RA_method No_RA   --dataset Hotpot   >/dev/null 2>&1 &
nohup  python run.py --ID Hotpot_1  --gpu 3 --RA_method Only_RA --dataset Hotpot   >/dev/null 2>&1 &

nohup  python run.py --ID Hotpot_2  --gpu 6 --RA_method MI_RA   --dataset Hotpot --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
nohup  python run.py --ID Hotpot_3  --gpu 6 --RA_method Gate_RA   --dataset Hotpot --gate_weight_0 2.5 --gate_weight_1 1  >/dev/null 2>&1 &
nohup  python run.py --ID Hotpot_4  --gpu 7 --RA_method Gate_MI_RA    --dataset Hotpot --gate_weight_0 2 --gate_weight_1 1 --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &


===================================================================================================================================================================================================
# new TriviaQA

nohup  python run.py --ID TriviaQA_0  --gpu 4 --RA_method No_RA   --dataset TriviaQA   >/dev/null 2>&1 &
nohup  python run.py --ID TriviaQA_1  --gpu 4 --RA_method Only_RA --dataset TriviaQA   >/dev/null 2>&1 &

nohup  python run.py --ID TriviaQA_2  --gpu 6 --RA_method MI_RA   --dataset TriviaQA --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &
nohup  python run.py --ID TriviaQA_3  --gpu 6 --RA_method Gate_RA   --dataset TriviaQA --gate_weight_0 2.5 --gate_weight_1 1  >/dev/null 2>&1 &
nohup  python run.py --ID TriviaQA_4  --gpu 7 --RA_method Gate_MI_RA    --dataset TriviaQA --gate_weight_0 2 --gate_weight_1 1 --quantile_num 0.7  --loss_list kl_soft+kl_hard+len_penalty --len_penalty_weight 1  >/dev/null 2>&1 &







