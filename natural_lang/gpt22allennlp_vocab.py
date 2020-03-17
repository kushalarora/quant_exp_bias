import json
vocab_list = []
vocab_dict = {}
with open('vocab.json') as vocab_json, \
    open('target_tokens.txt', 'w') as target_tokens:
    
    vocab_dict=json.load(vocab_json)
    vocab_list=['']*len(vocab_dict)
    for word,index in vocab_dict.items():
        vocab_list[index] = word
    for word in vocab_list:
        print(word, file=target_tokens)

