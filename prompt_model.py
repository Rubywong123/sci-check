import argparse
import os
import openai
import pandas as pd

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

import galai as gal
def get_response(prompt, args):
    if 'gpt' in args.model:
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

    else:
        # load galactica model
        # TODO: setup GPU
        model = gal.load_model(args.model.split('-')[-1])
        # nucleus sampling
        # Question: beam = 1 --> greedy sampling?
        response = model.generate(prompt, max_new_tokens=10, top_p = args.top_p)
        print(response)
        return response
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='choose model to prompt', choices=['gpt-3.5-turbo', 'galactica-mini', 'galactica-base'
                                                                           'galactica-standard', 'galactica-large', 'galactica-huge']
                                                                , default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default = 1.0)

    args = parser.parse_args()

    # read prompts
    prompt_df = pd.read_csv('data/Oracle_prompts.csv')

    # get responses
    response_df = pd.DataFrame(columns=['claim_id', 'doc_id', 'response'])

    for i, row in tqdm(prompt_df.iterrows()):
        prompt = row['prompt']
        response = get_response(prompt, args)
        response_df.loc[len(response_df)] = {'claim_id': row['claim_id'], 'doc_id': row['doc_id'], 'response': response}
    
    # store responses
    response_df.to_csv('data/{}_out.csv'.format(args.model))