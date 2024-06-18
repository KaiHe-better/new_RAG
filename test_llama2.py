# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from tqdm import tqdm

iter_count = 10
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

system_prompt = "You are a helpful assistant. Complete the sentence. Use only 10 words"
user_prompt = "JFK was the"
prompt_template = f"[INST] <<SYS>>\n {system_prompt} \n<</SYS>>\n\n {user_prompt}  [/INST]"

input_ids = tokenizer.encode(prompt_template, return_tensors="pt")
start_time = time.time()
for i in tqdm(range(iter_count)):
    output = model.generate(input_ids, max_new_tokens=20)

end_time = time.time()


print(tokenizer.decode(output[0], skip_special_tokens=True))
print(f"Time per generation : {(end_time - start_time)/iter_count} seconds")

