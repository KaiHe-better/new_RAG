import os
import yaml
import sys
import argparse
from dotenv import load_dotenv

parser = argparse.ArgumentParser()

# system settings
# parser.add_argument("--config", type=str, default="llama3-8b_USMLE_MI_RA.yaml", help="Path to the config file")
parser.add_argument('--gpu', default="6", type=str, help='gpu device numbers')
parser.add_argument("--test_code_flag", type=bool, default=False, help="if retrieval augmented")
parser.add_argument('--ID', type=str, default='7', help='run ID')
parser.add_argument('--seed', default=42, help='trandom seed')
parser.add_argument('--num_workers', default=48, type=int, help='data_loader_work')
parser.add_argument("--loading_ckpt_path", type=str, default=None, help="loading_ckpt_path, None ")
# In config
parser.add_argument("--RA_method", type=str,  default="Gate_RA", choices=["No_RA", "Only_RA", "Gate_RA", "MI_RA", "Gate_MI_RA"], help="RA_method")
parser.add_argument("--LLM", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="[meta-llama/Meta-Llama-3-8B-Instruct, meta-llama/Llama-2-7b-chat-hf]")  
# train
parser.add_argument('--dataset', type=str, default="PopQA", choices=["USMLE", "MedMCQA", "HEADQA", "PopQA", "Hotpot", "WebQA", "TriviaQA", "NQ"], help='train_file_path')
parser.add_argument('--train_batch_size', type=int, default=8, help='train_batch_size')
parser.add_argument('--test_batch_size', type=int, default=8, help='train_batch_size')
parser.add_argument('--accumulation_steps', type=int, default=1, help='accumulation_steps')
parser.add_argument('--demonstration', type=bool, default=False, help='in_context learning')
parser.add_argument('--demons_cnt', type=int, default=1, help='demonstration number')
parser.add_argument('--l2_coef', type=float, default=0, help='l2')
parser.add_argument('--train_eval', type=int, default=500, help='train_eval')
parser.add_argument('--total_step', type=int, default=25000, help='total_step')
parser.add_argument('--gate_weight_0', type=float, default=1, help='gate_weight_0')
parser.add_argument('--gate_weight_1', type=float, default=2, help='gate_weight_1')
# lr
parser.add_argument('--lr', type=float, default=1e-4, help='lr')
parser.add_argument('--init_lr_num', type=int, default=500, help='init_lr_num')
parser.add_argument('--lr_decay', type=float, default=0.9, help='lr_decay')
parser.add_argument('--lr_decay_interval', type=int, default=400, help='lr_decay_interval')
# MI model parameters
parser.add_argument("--num_layers", type=int,  default=1, help="num_layers")
parser.add_argument('--d_model', type=int, default=768, help='MI_learner dim')
parser.add_argument('--dim_feedforward', type=int, default=2048, help='MI_learner linear dim')
parser.add_argument('--layer_norm_eps', type=float, default=1e-5, help='MI_learner dim')
parser.add_argument('--nhead', type=int, default=8, help='MI_learner nhead')
parser.add_argument('--dropout', type=float, default=0.1, help='MI_learner dropout')
# loss
parser.add_argument('--loss_list', type=str, default="kl_soft+kl_hard", help='kl_soft+kl_hard+len_penalty')
parser.add_argument('--len_penalty_weight', type=float, default=10, help='soft_weight')
parser.add_argument('--soft_weight', type=float, default=1, help='soft_weight')
parser.add_argument('--hard_weight', type=float, default=1, help='hard_weight')
# decoding
parser.add_argument('--do_sample', type=bool, default=True, help='do_sample')
parser.add_argument("--temperature", type=float, default=1e-9, help="Temperature for decoding")
parser.add_argument("--top_p", type=float, default=0, help="Nucleus sampling top-p")
parser.add_argument("--max_new_tokens", type=int, default=40, help="Max number of new tokens to generate in one step， popqa=40")
# my_retrieval
parser.add_argument('--infer_add_gold_retrieval', type=bool, default=False, help='max_document_num')
parser.add_argument('--multi_query', type=bool, default=False, help='multi_query, using open AI')
parser.add_argument('--rewrite_num', type=int, default=1, help='1 or 2')
parser.add_argument('--chunk_size', type=int, default=512, help='chunk_sizen, not token length')
parser.add_argument('--chunk_overlap', type=int, default=20, help='chunk_sizen, not token length')
parser.add_argument('--if_hierarchical_retrieval', type=bool, default=True, help='if_hierarchical_retrieval')
parser.add_argument('--hierarchical_ratio', type=float, default=1.4, help='hierarchical_ratio, 1-2')
parser.add_argument('--quantile_num', type=float, default=1, help='quantile_num, 0.8-1.1')
# retriever
parser.add_argument("--n_docs", type=int, default=10, help="Number of documents to retrieve per questions")
parser.add_argument("--model_name_or_path", type=str,  default="facebook/contriever-msmarco", choices=["facebook/dragon-plus-query-encoder", "facebook/contriever-msmarco"], help="triever to use")
parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
parser.add_argument("--passages", type=str, default="datasets/Retrieval_corpus/enwiki_2020_dec_intro_only.jsonl", help="Path to passages (.tsv file)")
parser.add_argument("--passages_embeddings", type=str, default="datasets/Retrieval_corpus/enwiki_dec_2020_contriever_intro/*", help="Glob path to encoded passages")

parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
parser.add_argument("--normalize_text", action="store_true", help="normalize text")

parser.add_argument("--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists")
parser.add_argument("--no_fp16", type=bool, default=True, help="inference in fp32")
parser.add_argument("--projection_size", type=int, default=768)
parser.add_argument("--n_subquantizers", type=int, default=0, help="Number of subquantizer used for vector quantization, if 0 flat index is used")
parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
parser.add_argument("--indexing_batch_size", type=int, default=30000000, help="Batch size of the number of passages indexed, 30000000")
                                                               
# parser.add_argument("--lang", nargs="+")
# parser.add_argument("--dataset", type=str, default="none")
# parser.add_argument("--validation_workers", type=int, default=32, help="Number of parallel processes to validate results")
# parser.add_argument("--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix")

args = parser.parse_args()
# args.config = "configs/"+args.dataset+ "/"+args.config
# config = yaml.safe_load(open(args.config)) 
# parser.set_defaults(**config)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TOKENIZERS_PARALLELISM"] = "true"

load_dotenv(".env")
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

from dataloader.data_loader import get_loader  
from trainer import My_Trainer
import torch
from utils.utils import load_LLM, get_logger, make_log_dir, seed_everything
from models.my_model import My_MI_learner, My_gate
from src.passage_retrieval import Retriever

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
args.device = device

seed_everything(int(args.seed))

args.prompt_file = "prompts/" + args.dataset + ".json"

if args.dataset in ["USMLE", "MedMCQA", "HEADQA"]:
    args.max_new_tokens = 1
else:
    args.max_new_tokens = 40

dir_path = make_log_dir()
args.dir_path = dir_path
args.print_logger = get_logger(dir_path, "print")


def custom_excepthook(exc_type, exc_value, exc_traceback):
    args.print_logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
sys.excepthook = custom_excepthook

args.print_logger.info("**************** Configuration **************** ")
for k in args.__dict__:
    args.print_logger.info(f"{k}: {args.__dict__[k]}")
args.print_logger.info("**************** Configuration **************** \n\n")



def main(args):
   
    if args.RA_method == "No_RA":
        retriever = None
    else:    
        args.print_logger.info("Loading retriever ...")
        
        if args.test_code_flag==True:
            args.passages_embeddings = "datasets/Retrieval_corpus/enwiki_dec_2020_contriever_intro/passages_00"
            args.train_eval=2

        retriever = Retriever(args)
        retriever.setup_retriever()

    # retriever =None

    LLM, LLM_tokenizer = load_LLM(args)
    train_data_loader, dev_data_loader, test_data_loader = get_loader(args, LLM_tokenizer)
    
    if args.RA_method in ["Gate_RA", "Gate_MI_RA"]:
        my_gate = My_gate(args)
    else:
        my_gate = None

    if args.RA_method in ["Gate_MI_RA", "MI_RA"]: 
        if args.LLM == "meta-llama/Meta-Llama-3-8B-Instruct":
            MI_learner = My_MI_learner(args, LLM_tokenizer.vocab_size+len(LLM_tokenizer.added_tokens_encoder) if args.LLM != "chatGPT" else 32000)
        else:
            MI_learner = My_MI_learner(args, LLM_tokenizer.vocab_size if args.LLM != "chatGPT" else 32000)
    else:
        MI_learner = None

    if args.loading_ckpt_path is not None:
        args.print_logger.info(f"loading ckpt from {args.loading_ckpt_path} ! \n=================\n")
        MI_learner.load_state_dict(torch.load(args.loading_ckpt_path))

    trainer = My_Trainer(args, MI_learner, my_gate, LLM, LLM_tokenizer, device, retriever)
    
    if args.RA_method in ["Gate_RA", "Gate_MI_RA", "MI_RA"]:
        # trainer.train_proc(train_data_loader, dev_data_loader)
        trainer.train_proc(train_data_loader, test_data_loader)
    elif args.RA_method in ["No_RA", "Only_RA"]:
        test_performce, test_performce_in, all_test_predictions, all_test_input_list, all_test_answers = trainer.test_proc(test_data_loader)  

        test_result_logger = get_logger(args.dir_path, "test_result")
        for batch_pred, batch_input, batch_answer in zip(all_test_predictions, all_test_input_list, all_test_answers):
            trainer.recored_res(batch_pred, batch_input, batch_answer, training_flag=False, record_flag=True) 

    else:
        raise Exception("wrong RA_method")



if __name__ == "__main__":
    main(args)
