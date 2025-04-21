import json

tokenizer_path = '/home/dist/Llama-2-7b/tokenizer.json'

with open(tokenizer_path, 'r') as f:
    con = json.load(f)
    
print(con.keys())

vocab = con['model']['vocab']
with open('aquila/tokenizer_llama/vocab.json', 'w') as f:
    json.dump(vocab, f)
    
with open('aquila/tokenizer_llama/merges.txt', 'w') as f:
    f.write('\n'.join(con['model']['merges']) + '\n')
    
with open('aquila/tokenizer_llama/special_tokens.txt', 'w') as f:
    f.write("<s>\n")
