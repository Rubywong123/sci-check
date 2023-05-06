import argparse
import os
import openai
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from transformers import AutoTokenizer, OPTForCausalLM

def GPT_response(prompt, args):
    if 'turbo' not in args.model:
        response = openai.Completion.create(
            model = args.model,
            prompt = prompt,
            temperature = args.temperature,
            top_p = args.top_p,
            max_tokens = 20,
        )

        print(response)
            
        return response.choices[0].text.strip()
    
    else:
        response = openai.ChatCompletion.create(
            model = args.model,
            messages = [{'role':'user', 'content':prompt}],
            temperature = args.temperature,
            top_p = args.top_p,
            max_tokens = 20,
        )

        print(response)
            
        return response.choices[0]['message']['content'].strip()
def galactica_response(prompt, model: OPTForCausalLM, tokenizer, device, args):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    outputs = model.generate(
        input_ids, 
        max_new_tokens=20, 
        temperature=args.temperature,
        top_p = args.top_p,
    )
    print(tokenizer.decode(outputs[0]))
    return tokenizer.decode(outputs[0])

if __name__ == '__main__':
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='choose model to prompt', choices=['gpt-3.5-turbo', 'galactica-mini', 'galactica-base'
                                                                           'galactica-standard', 'galactica-large', 'galactica-huge']
                                                                , default='galactica-mini')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default = 1.0)

    args = parser.parse_args()

    # read prompts
    prompt_df = pd.read_csv('data/Oracle_prompts.csv')

    # get responses
    response_df = pd.DataFrame(columns=['claim_id', 'doc_id', 'response'])

    if 'galactica' in args.model:
        print("=================LOADING MODEL====================")
        tokenizer = AutoTokenizer.from_pretrained('./checkpoints/{}'.format(args.model), torch_dtype=torch.float16)
        model = OPTForCausalLM.from_pretrained('./checkpoints/{}'.format(args.model), torch_dtype=torch.float16)

        if torch.cuda.is_available():   #GPU running
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        model.to(device)
        
        

    for i, row in tqdm(prompt_df.iterrows()):
        prompt = row['prompt']
        if 'galactica' not in args.model:
            response = GPT_response(prompt, args)
        else:
            response = galactica_response(prompt, model, tokenizer, device, args)
        response_df.loc[len(response_df)] = {'claim_id': row['claim_id'], 'doc_id': row['doc_id'], 'response': response}
    
    # store responses
    response_df.to_csv('data/{}_out.csv'.format(args.model))