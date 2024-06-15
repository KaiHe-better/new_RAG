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

nohup python run.py --ID USMLE_0 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --gate_weight 1  --len_penalty_weight 4 >/dev/null 2>&1 &
2024-05-07 23:29:31,590 test: acc 49.49, f1 47.4, precision 48.63, recall 48.87, old_doc_len:2480.0533751962325, new_doc_len:872.234693877551, hallucination: 0.08 
2024-05-07 23:29:31,590 cost_time: 29.669101238250732 , gate_res_list: 0.648075412411626, 825 / 1273 

nohup python run.py --ID USMLE_3 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --gate_weight 2  --len_penalty_weight 4 >/dev/null 2>&1 &
2024-05-08 03:41:56,168 test: acc 49.49, f1 47.37, precision 48.63, recall 48.82, old_doc_len:2480.0533751962325, new_doc_len:892.6883830455259, hallucination: 0.08 
2024-05-08 03:41:56,168 cost_time: 50.711981503168744 , gate_res_list: 0.6543597800471328, 833 / 1273 

nohup python run.py --ID USMLE_0 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard             --soft_weight 1 --hard_weight 1  >/dev/null 2>&1 &
2024-05-09 08:05:25,223 test: acc 49.57, f1 47.46, precision 48.66, recall 48.36, old_doc_len:2480.0533751962325, new_doc_len:808.4748822605966, hallucination: 0.0 
2024-05-09 08:05:25,223 cost_time: 29.41408345301946 , gate_res_list: 0.6575019638648861, 837 / 1273 

nohup python run.py --ID USMLE_1 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --len_penalty_weight 1 >/dev/null 2>&1 &
2024-05-10 03:37:33,006 test: acc 49.33, f1 47.21, precision 48.49, recall 48.41, old_doc_len:2480.0533751962325, new_doc_len:951.1287284144427, hallucination: 0.0 
2024-05-10 03:37:33,006 cost_time: 44.86397190888723 , gate_res_list: 0.6622152395915161, 843 / 1273 

nohup python run.py --ID USMLE_2 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --len_penalty_weight 2 >/dev/null 2>&1 &
2024-05-08 19:45:30,506 test: acc 49.49, f1 47.37, precision 48.57, recall 48.78, old_doc_len:2480.0533751962325, new_doc_len:846.4073783359497, hallucination: 0.0 
2024-05-08 19:45:30,506 cost_time: 32.293471018473305 , gate_res_list: 0.6567164179104478, 836 / 1273 

nohup python run.py --ID USMLE_3 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --len_penalty_weight 3 >/dev/null 2>&1 &
2024-05-08 20:41:18,559 test: acc 49.65, f1 47.55, precision 48.72, recall 48.99, old_doc_len:2480.0533751962325, new_doc_len:864.5368916797488, hallucination: 0.08 
2024-05-08 20:41:18,559 cost_time: 35.15572950442632 , gate_res_list: 0.6567164179104478, 836 / 1273 



# new llama2-7b_USMLE_MI_RA  --n_docs 5

nohup python run.py --ID USMLE_0 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 5 --loss_list kl_soft+kl_hard             --soft_weight 1 --hard_weight 1  >/dev/null 2>&1 &
2024-05-11 14:14:03,076 test: acc 50.27, f1 48.28, precision 49.24, recall 48.98, old_doc_len:1252.7605965463108, new_doc_len:542.4568288854003, hallucination: 0.0 
2024-05-11 14:14:03,077 cost_time: 26.357947031656902 , gate_res_list: 0.6614296936370778, 842 / 1273 

nohup python run.py --ID USMLE_1 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 5 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --len_penalty_weight 5 >/dev/null 2>&1 &
2024-05-10 23:00:17,128 test: acc 50.2, f1 48.16, precision 49.14, recall 48.92, old_doc_len:1252.7605965463108, new_doc_len:536.6758241758242, hallucination: 0.0 
2024-05-10 23:00:17,128 cost_time: 29.71108977397283 , gate_res_list: 0.6575019638648861, 837 / 1273 

nohup python run.py --ID USMLE_2 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 5 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --len_penalty_weight 8 >/dev/null 2>&1 &
2024-05-10 21:50:18,816 test: acc 50.2, f1 48.16, precision 49.14, recall 48.92, old_doc_len:1252.7605965463108, new_doc_len:536.6758241758242, hallucination: 0.0 
2024-05-10 21:50:18,816 cost_time: 26.213221351305645 , gate_res_list: 0.6575019638648861, 837 / 1273 

nohup python run.py --ID USMLE_3 --gpu 7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --n_docs 5 --loss_list kl_soft+kl_hard+len_penalty --soft_weight 1 --hard_weight 1 --len_penalty_weight 10 >/dev/null 2>&1 &
2024-05-10 22:30:25,900 test: acc 50.2, f1 48.16, precision 49.14, recall 48.92, old_doc_len:1252.7605965463108, new_doc_len:536.6758241758242, hallucination: 0.0 
2024-05-10 22:30:25,900 cost_time: 29.351856935024262 , gate_res_list: 0.6575019638648861, 837 / 1273 


# new llama3-8b_USMLE_MI_RA 


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
