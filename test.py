

from vllm import LLM, SamplingParams

# model = LLM("selfrag/selfrag_llama2_7b", dtype="half")
model = LLM("mistralai/Mistral-7B-Instruct-v0.2", dtype="half")
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)

def format_prompt(input, paragraph=None):
  prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)
  if paragraph is not None:
    prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
  return prompt

# query_1 = "Leave odd one out: twitter, instagram, whatsapp."
# query_2 = "Can you tell me the difference between llamas and alpacas?"
# queries = [query_1, query_2]

# # for a query that doesn't require retrieval
# preds = model.generate([format_prompt(query) for query in queries], sampling_params)
# for pred in preds:
#   print("Model prediction: {0}".format(pred.outputs[0].text))




query_3 = "Can you tell me the difference between llamas and alpacas?"


from src.passage_retrieval import Retriever
retriever = Retriever({})
passages = "datasets/Retrieval_corpus/enwiki_2020_dec_intro_only.jsonl"
passages_embeddings = "datasets/Retrieval_corpus/enwiki_dec_2020_contriever_intro/*"
retriever.setup_retriever_demo("facebook/contriever-msmarco", passages, passages_embeddings,  n_docs=5, save_or_load_index=False)
retrieved_documents = retriever.search_document_demo(query_3, 5)
prompts = [format_prompt(query_3, doc["title"] +"\n"+ doc["text"]) for doc in retrieved_documents]
preds = model.generate(prompts, sampling_params)
top_doc = retriever.search_document_demo(query_3, 1)[0]
print("Reference: {0}\n\n\n Model prediction: {1}  finish!".format(top_doc["title"] + "\n" + top_doc["text"], preds[0].outputs[0].text))