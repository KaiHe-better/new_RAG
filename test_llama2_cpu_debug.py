# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

import time
from tqdm import tqdm

iter_count = 1
batch_size = 1
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

model.to("cpu")
model.eval()

generation_config = GenerationConfig(
            max_new_tokens=10,
            pad_token_id = tokenizer.eos_token_id,
            do_sample=False,
            num_return_sequences=1,
            # temperature=self.args.temperature,
            # top_p=self.args.top_p,
            # length_penalty=self.args.length_penalty,
            # num_beams=self.args.num_beams,
            return_dict_in_generate=True,
            output_scores=True,
        )

system_prompt = "You are a helpful assistant. Complete the sentence. Use only 10 words"
user_prompt = "JFK was the"
prompt_template = f"[INST] <<SYS>>\n {system_prompt} \n<</SYS>>\n\n {user_prompt}  [/INST]"
input_list = []
for i in range(batch_size):
    input_list.append(prompt_template)

input_ids = tokenizer(input_list, return_tensors="pt")
#input_ids = input_ids.to("hpu")

start_time = time.time()
for i in tqdm(range(iter_count)):
    input_ids['input_ids'] = torch.randint(1,29999,input_ids['input_ids'].shape)
    input_ids['input_ids'][0][0]=torch.ones(1)
    input_ids = input_ids.to("cpu")
    output = model.generate(**input_ids, generation_config=generation_config)

end_time = time.time()
print(tokenizer.decode(output["sequences"][0], skip_special_tokens=True))
print(f"Time per generation : {(end_time - start_time)/iter_count} seconds")

