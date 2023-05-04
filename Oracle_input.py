import jsonlines
from tqdm import tqdm
import pandas as pd
import random
import numpy as np

if __name__ == '__main__':

    corpus_df = pd.DataFrame(columns = ['doc_id', 'title', 'abstract', 'metadata'])
    with open('data/corpus_candidates.jsonl', 'r', encoding = 'utf-8') as f:
        for item in jsonlines.Reader(f):
            corpus_df.loc[len(corpus_df)] = item

    df = pd.DataFrame(columns = ['id', 'claim', 'evidence'])
    with open('data/claims.jsonl', 'r', encoding = 'utf-8') as f:
        for item in jsonlines.Reader(f):
            df.loc[len(df)] = item
    
    out_df = pd.DataFrame(columns = ['id', 'doc_id'])

    #fix random seed
    random.seed(42)

    for index, row in tqdm(df.iterrows()):
        
        selected_papers = list(map(int, row['evidence'].keys()))


        # TODO: add negative samples
        if len(selected_papers) < 5:
            cnt = 5 - len(selected_papers)
            # random choice from corpus
            while cnt > 0:
                idx = random.randint(0, corpus_df.shape[0])
                doc_id = corpus_df.iloc[idx]['doc_id']
                if doc_id not in selected_papers:
                    selected_papers.append(doc_id)
                    cnt -= 1
                

        out_df.loc[len(out_df)] = {'id': row['id'], 'doc_id': selected_papers}
        

    out_df.to_csv('Oracle_input.csv')

