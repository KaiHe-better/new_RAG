
nohup python run.py --ID USMLE_00 --gpu 4 --config mistral_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft+kl_hard --soft_weight 0.5 --hard_weight 0.5 >/dev/null 2>&1 & 

nohup python run.py --ID USMLE_012 --gpu 4 --config mistral_USMLE_MI_RA.yaml --dataset USMLE --n_docs 20 --quantile_num 0.8 --loss_list kl_soft+kl_hard  >/dev/null 2>&1 &  
nohup python run.py --ID USMLE_013 --gpu 6 --config mistral_USMLE_MI_RA.yaml --dataset USMLE --n_docs 20 --quantile_num 0.5 --loss_list kl_soft+kl_hard  >/dev/null 2>&1 &  


nohup python run.py --ID USMLE_02 --gpu 6 --config mistral_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --loss_list kl_soft >/dev/null 2>&1 &  
nohup python run.py --ID USMLE_03 --gpu 6 --config mistral_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 --quantile_num 0.8 --loss_list kl_hard >/dev/null 2>&1 &  

nohup python run.py --ID USMLE_6 --gpu 5 --config mistral_USMLE_RA.yaml --dataset USMLE  >/dev/null 2>&1 &  


nohup python run.py --ID USMLE_0 --gpu 4 --config llama3-8b_USMLE_RA.yaml  --dataset USMLE --n_docs 10 >/dev/null 2>&1 &  
nohup python run.py --ID USMLE_1 --gpu 6 --config llama3-8b_USMLE_RA.yaml  --dataset USMLE --n_docs 20 >/dev/null 2>&1 &  
