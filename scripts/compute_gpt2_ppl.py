# from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# from pytorch_transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel 

import sys
import torch

import pdb;pdb.set_trace()
if len(sys.argv) > 2:
    model_name = sys.argv[2]
else:
    model_name = 'gpt2'
# model_name = 'openai-gpt'
# tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
# model = OpenAIGPTLMHeadModel.from_pretrained(model_name).to(torch.cuda.current_device())

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(torch.cuda.current_device())
loss = 0.
count = 0
with open(sys.argv[1]) as f:
    for i, line in enumerate(f):
        tokens = tokenizer.tokenize(line)
        if len(tokens) < 2:
            continue

        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]).to(torch.cuda.current_device())
        with torch.no_grad():
            loss += len(tokens) * model(tensor_input, labels=tensor_input)[0]
            count += len(tokens)
        print('{0}\r'.format(i), end='\r')

    ppl = torch.exp(loss/count) 
    print(ppl)
