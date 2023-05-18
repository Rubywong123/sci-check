import jsonlines
from tqdm import tqdm
import pandas as pd
import random
import numpy as np

if __name__ == '__main__':
    enum = {"SUPPORT": 1, 'CONTRADICT': 2}

    corpus_df = pd.DataFrame(columns = ['doc_id', 'title', 'abstract', 'metadata'])
    docid2index = {}
    with open('data/scifact/corpus.jsonl', 'r', encoding = 'utf-8') as f:
        for i, item in tqdm(enumerate(jsonlines.Reader(f))):
            corpus_df.loc[len(corpus_df)] = item
            docid2index[item['doc_id']] = i

    df = pd.DataFrame(columns = ['id', 'claim', 'evidence'])
    with open('data/scifact/claims_train_cited.jsonl', 'r', encoding = 'utf-8') as f:
        for item in jsonlines.Reader(f):
            df.loc[len(df)] = item
    prompt_template = '''{}</s>{}'''
    out_df = pd.DataFrame(columns = ['id', 'doc_id', 'document', 'label'])

    #fix random seed
    random.seed(42)

    for index, row in tqdm(df.iterrows()):
        
        selected_papers = list(map(int, row['evidence'].keys()))

        for doc_id in selected_papers:
            paper = corpus_df.iloc[docid2index[doc_id]]
            out_df.loc[len(out_df)] = {'id': row['id'], 'doc_id': doc_id, 
                                       'document': prompt_template.format(row['claim'], paper['title'] + ' ' + ' '.join(paper['abstract'])),
                                       'label': enum[row['evidence'][str(doc_id)][0]['label']]}

        # TODO: add negative samples
        if len(selected_papers) < 10:
            cnt = 10 - len(selected_papers)
            # random choice from corpus
            while cnt > 0:
                idx = random.randint(0, corpus_df.shape[0]-1)
                doc_id = corpus_df.iloc[idx]['doc_id']
                if doc_id not in selected_papers:
                    selected_papers.append(doc_id)
                    
                    paper = corpus_df.iloc[docid2index[doc_id]]
                    out_df.loc[len(out_df)] = {'id': row['id'], 'doc_id': doc_id,
                                                'document': prompt_template.format(row['claim'], paper['title'] + ' ' + ' '.join(paper['abstract'])),
                                                'label': 0}

                    cnt -= 1
    
    out_df.to_csv('class_ft_data.csv')

