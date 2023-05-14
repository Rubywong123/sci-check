import argparse
import os
import openai
import pandas as pd
import torch
# from dotenv import load_dotenv
from tqdm import tqdm
import jsonlines
from transformers import AutoTokenizer, OPTForCausalLM

import ast

def parse_list(s):
    return ast.literal_eval(s) if isinstance(s, str) else s

prompt_template = '''
Is the Document relevant to the claim? Answer Yes or No. Claim: {} Document:{} Answer: '''

def get_components(input: str):
    body = input[input.index('Title'): input.index('Question: ')]
    

    claim = input[input.index('Question: ') + 10: input.index('True')]


    return body, claim

model = 'galactica-base'

df = pd.read_csv('data/Oracle_prompts.csv')

print("=================LOADING MODEL====================")
        

device = torch.device('cpu')

if torch.cuda.is_available():   #GPU running
    tokenizer = AutoTokenizer.from_pretrained('./checkpoints/{}'.format(model), torch_dtype=torch.float16)
    model = OPTForCausalLM.from_pretrained('./checkpoints/{}'.format(model), torch_dtype=torch.float16)
    device = torch.device('cuda')


print(device)
model.to(device)

claim_df = pd.DataFrame(columns = ['id', 'claim', 'evidence'])
with open('data/claims.jsonl', 'r', encoding = 'utf-8') as f:
    for item in jsonlines.Reader(f):
        claim_df.loc[len(claim_df)] = item

chosen_num = 0
chosen_correct_num = 0

claim_idx = 0
for i, row in tqdm(df.iterrows()):
        
    prompt = row['prompt']
    claim_id = row['claim_id']

    gold_claim_id = claim_df.iloc[claim_idx]['id']

    if claim_id != gold_claim_id:
        claim_idx += 1

    gold_doc_list = claim_df.iloc[claim_idx]['evidence'].keys()
    
    body, claim = get_components(prompt)

    prompt = prompt_template.format(body, claim)

    # print(prompt)

    input_ids = tokenizer(prompt, max_length = 2020, truncation=True, return_tensors='pt').input_ids.to(device)

    outputs = model.generate(
        input_ids, 
        max_new_tokens=20,
        temperature= 1,
        return_dict_in_generate=True,
        output_scores=True,
        # num_beams=2,
        # force_words_ids= [[34960, 6920]]           
    )

    # print(outputs)

    response = tokenizer.decode(outputs['sequences'][0])

    # "True" and "False".
    print(response[len(prompt):])
    Yes_score = outputs.scores[0][0][34960]
    No_score = outputs.scores[0][0][6920]
    print("Yes score: ", Yes_score)
    print("No score: ", No_score)
    print("Max score: ", outputs.scores[0][0].argmax(), tokenizer.decode(outputs.scores[0][0].argmax()))
    with open('filter_res.log', 'a', encoding='utf-8') as f:
        f.write("Prompt {}\n".format(i))
        f.write(response[len(prompt):] + '\n')
        f.write("Yes score: " + str(Yes_score))
        f.write('\n')
        f.write("No score: "+ str(No_score))
        f.write('\n')

    if 'Yes' in response[len(prompt):len(prompt) + 5] or Yes_score > No_score:
        chosen_num += 1
        if str(row['doc_id']) in gold_doc_list:
            chosen_correct_num += 1

    

print("CHOSEN: ", chosen_num)
print("CHOSEN_CORRECT_NUM: ", chosen_correct_num)



    




