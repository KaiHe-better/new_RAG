
nohup python run.py --ID USMLE_0 --gpu 4 --config mistral_USMLE_RA.yaml --dataset USMLE --n_docs 1 >/dev/null 2>&1 & 
nohup python run.py --ID USMLE_1 --gpu 6 --config mistral_USMLE_RA.yaml --dataset USMLE --n_docs 5 >/dev/null 2>&1 &  
nohup python run.py --ID USMLE_2 --gpu 7 --config mistral_USMLE_RA.yaml --dataset USMLE --n_docs 10 >/dev/null 2>&1 &  


nohup python run.py --ID USMLE_3 --gpu 4 --config mistral_USMLE_MI_RA.yaml --dataset USMLE --n_docs 1 >/dev/null 2>&1 & 
nohup python run.py --ID USMLE_4 --gpu 6 --config mistral_USMLE_MI_RA.yaml --dataset USMLE --n_docs 5 >/dev/null 2>&1 &  
nohup python run.py --ID USMLE_5 --gpu 7 --config mistral_USMLE_MI_RA.yaml --dataset USMLE --n_docs 10 >/dev/null 2>&1 &  
