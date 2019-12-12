# from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# from pytorch_transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel 

import sys
import torch
import math

import torch.nn.functional as F

import pdb;pdb.set_trace()
if len(sys.argv) > 2:
    model_name = sys.argv[2]
else:
    model_name = 'gpt2'
# model_name = 'openai-gpt'
# tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
# model = OpenAIGPTLMHeadModel.from_pretrained(model_name).to(torch.cuda.current_device())

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)#.to(torch.cuda.current_device())
model.eval()
loss = 0.
count = 0
sequences = []
with open(sys.argv[1]) as f:
    for i, line in enumerate(f):
        tokens = tokenizer.tokenize(line)
        if len(tokens) < 2:
            continue
        
        sequences.append(tokens)
    #     tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]).to(torch.cuda.current_device())
    #     with torch.no_grad():
    #         loss += len(tokens) * model(tensor_input, labels=tensor_input)[0]
    #         count += len(tokens)
    #     print('{0}\r'.format(i), end='\r')

    # ppl = torch.exp(loss/count) 
    # print(ppl)

loss2 = 0.
count2 = 0.
max_len = max([len(sequence) for sequence in sequences])
ids = [tokenizer.convert_tokens_to_ids(sequence) + [tokenizer.eos_token_id] * (max_len - len(sequence)) for sequence in sequences]
tensor_input = torch.tensor(ids)#.to(torch.cuda.current_device())
attention_mask = (tensor_input != tokenizer.eos_token_id).float()#.to(torch.cuda.current_device())
batch_size = 100
import pdb;pdb.set_trace()

for i in range(0, len(sequences), batch_size):
    inp = tensor_input[i:i+batch_size] if i + batch_size < len(sequences) else tensor_input[i:len(sequences)]
    mask = attention_mask[i:i+batch_size] if i + batch_size < len(sequences) else attention_mask[i:len(sequences)]
    
    with torch.no_grad():

        results =  model(inp, labels=inp, attention_mask=mask)
        logits = results[1]
        labels = inp

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_batch_seq = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                            shift_labels.view(-1),
                                            ignore_index = -1, reduction='none').view(batch_size, -1)

        loss_batch_seq *=mask[:, 1:]
        seq_sizes = mask.sum(dim=-1)

        loss_batch = loss_batch_seq.sum(dim=-1)/seq_sizes

        for j, sequence in enumerate(sequences[i:i+batch_size]):
            loss2 += len(sequence) * loss_batch[j].item()
            count2 += len(sequence)

    print('{0}, {1}\r'.format(i, math.exp(loss2/count2)), end='\r')
print(math.exp(loss2/count2))

