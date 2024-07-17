from dataloader.usmle_loader import get_loader_USMLE
from dataloader.ottqa_loader import get_loader_OTTQA
from dataloader.medmcqa_loader import get_loader_MedMCQA
from dataloader.mmlu_loader import get_loader_MMLU 


from dataloader.popqa_loader import get_loader_PopQA 
from dataloader.hotpot_loader import get_loader_Hotpot 

from dataloader.webqa_loader import get_loader_WebQA 
from dataloader.nq_loader import get_loader_NQ
from dataloader.triviaqa_loader import get_loader_TriviaQA


def get_loader(args, tokenizer):
    # print ('Loading data...')
    args.print_logger.info(f"Loading data ...")
    train_data_loader, dev_data_loader, test_data_loader = "", "", ""

    if args.dataset == "USMLE":
        train_file_path = "datasets/Downstream/USMLE/questions/US/4_options/phrases_no_exclude_train.jsonl"
        dev_file_path = "datasets/Downstream/USMLE/questions/US/4_options/phrases_no_exclude_dev.jsonl"
        test_file_path = "datasets/Downstream/USMLE/questions/US/4_options/phrases_no_exclude_test.jsonl"

        rewrite_train_file_path = "datasets/Downstream/USMLE/questions/US/4_options/rewrite_USMLE_train.json"
        rewrite_dev_file_path = "datasets/Downstream/USMLE/questions/US/4_options/rewrite_USMLE_dev.json"
        rewrite_test_file_path = "datasets/Downstream/USMLE/questions/US/4_options/rewrite_USMLE_test.json"

        train_data_loader, dev_data_loader, test_data_loader, args = get_loader_USMLE(args, tokenizer, train_file_path, dev_file_path, test_file_path, 
                                                                                      rewrite_train_file_path, rewrite_dev_file_path, rewrite_test_file_path
                                                                                      ) 

    elif args.dataset == "MedMCQA":
        train_file_path = "datasets/Downstream/MedMCQA/train.json"
        dev_file_path = "datasets/Downstream/MedMCQA/dev.json"
        test_file_path = "datasets/Downstream/MedMCQA/test.json"

        rewrite_train_file_path = None
        rewrite_dev_file_path = "datasets/Downstream/MedMCQA/rewrite_MedMCQA_dev.json"
        rewrite_test_file_path = "datasets/MedMCQA/rewrite_MedMCQA_test.json"

        train_data_loader, dev_data_loader, test_data_loader, args = get_loader_MedMCQA(args, tokenizer, train_file_path, dev_file_path, test_file_path,
                                                                                        rewrite_train_file_path, rewrite_dev_file_path, rewrite_test_file_path,
                                                                                        ) 
    elif args.dataset == "HEADQA":
        train_file_path = "datasets/Downstream/HEADQA/train.json"
        dev_file_path = "datasets/Downstream/HEADQA/dev.json"
        test_file_path = "datasets/Downstream/HEADQA/test.json"

        rewrite_train_file_path = "datasets/Downstream/HEADQA/rewrite_HEADQA_train.json"
        rewrite_dev_file_path = "datasets/Downstream/HEADQA/rewrite_HEADQA_test.json"
        rewrite_test_file_path = "datasets/Downstream/HEADQA/rewrite_HEADQA_test.json"

        train_data_loader, dev_data_loader, test_data_loader, args = get_loader_HEADQA(args, tokenizer, train_file_path, dev_file_path, test_file_path,
                                                                                       rewrite_train_file_path, rewrite_dev_file_path, rewrite_test_file_path
                                                                                       ) 

    elif args.dataset == "PopQA":
        train_file_path = "datasets/Downstream/PopQA/popqa_train.jsonl"
        dev_file_path = "datasets/Downstream/PopQA/popqa_valid.jsonl"
        test_file_path = "datasets/Downstream/PopQA/popqa_test.jsonl"
                         
        train_data_loader, dev_data_loader, test_data_loader, args = get_loader_PopQA(args, tokenizer, train_file_path, dev_file_path, test_file_path) 

    elif args.dataset == "Hotpot":
        train_file_path = "datasets/Downstream/Hotpot/hotpot_train_v1.1.json"
        dev_file_path = "datasets/Downstream/Hotpot/hotpot_dev_fullwiki_v1.json"
        test_file_path = "datasets/Downstream/Hotpot/hotpot_dev_fullwiki_v1.json"
        # test_file_path = "datasets/Downstream/Hotpot/hotpot_test_fullwiki_v1.json"
                         
        train_data_loader, dev_data_loader, test_data_loader, args = get_loader_Hotpot(args, tokenizer, train_file_path, dev_file_path, test_file_path) 


    elif args.dataset == "WebQA":
        train_file_path = "datasets/Downstream/xRAG/WebQA/webq-train.jsonl"
        dev_file_path = "datasets/Downstream/xRAG/WebQA/webq-dev.jsonl"
        test_file_path = "datasets/Downstream/xRAG/WebQA/webq-test.jsonl"
                         
        train_data_loader, dev_data_loader, test_data_loader, args = get_loader_WebQA(args, tokenizer, train_file_path, dev_file_path, test_file_path) 


    elif args.dataset == "TriviaQA":
        # train_file_path = "datasets/Downstream/xRAG/TriviaQA/tqa-train.jsonl"
        # dev_file_path = "datasets/Downstream/xRAG/TriviaQA/tqa-dev.jsonl"
        # test_file_path = "datasets/Downstream/xRAG/TriviaQA/tqa-test.jsonl"
        
        train_file_path = "datasets/Downstream/Triviaqa/triviaqa-unfiltered/unfiltered-web-train.json"
        dev_file_path = "datasets/Downstream/Triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json"
        test_file_path = "datasets/Downstream/Triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json"

        train_data_loader, dev_data_loader, test_data_loader, args = get_loader_TriviaQA(args, tokenizer, train_file_path, dev_file_path, test_file_path) 

    elif args.dataset == "NQ":
        train_file_path = "datasets/Downstream/xRAG/NQ/nq-train.jsonl"
        dev_file_path = "datasets/Downstream/xRAG/NQ/nq-dev.jsonl"
        test_file_path = "datasets/Downstream/xRAG/NQ/nq-test.jsonl"
                         
        train_data_loader, dev_data_loader, test_data_loader, args = get_loader_NQ(args, tokenizer, train_file_path, dev_file_path, test_file_path) 


    else:
        raise Exception("Wrong dataset selected !")


    
    return train_data_loader, dev_data_loader, test_data_loader