import pandas as pd
from tqdm import tqdm
import jsonlines
import os
import pickle


def form_prompt(paper, claim):
    prompt_template = """Answer the Question according to the Abstract
    
Title: {}

Abstract: {}

Question: {} True, False, or Neither?

Answer: """
    abstract_counter = [x + 1 for x in range(len(paper["abstract"]))]
    abstract_sents = [
        f"[{counter}] {sentence}"
        for counter, sentence in zip(abstract_counter, paper["abstract"])
    ]
    abstract_sents = " ".join(abstract_sents)
    
    prompt = prompt_template.format(paper['title'], abstract_sents, claim['claim'])
    print(prompt)

    return prompt

if __name__ == '__main__':

    # column: 
    # 'id': claim_id
    # 'doc_id': list of ids of retrieved documents for current claim

    # get BM25 result
    df = pd.read_csv('BM25_result.csv')

    # use claim_id to get claim
    claim_df = pd.DataFrame(columns = ['id', 'claim', 'evidence'])
    with open('data/claims.jsonl', 'r', encoding = 'utf-8') as f:
        for item in jsonlines.Reader(f):
            claim_df.loc[len(claim_df)] = item

    # use doc_id to get abstract sentences
    doc_df = pd.DataFrame(columns = ['doc_id', 'title', 'abstract', 'metadata'])
    with open('data/corpus_candidates.jsonl', 'r', encoding = 'utf-8') as f:
        for item in jsonlines.Reader(f):
            doc_df.loc[len(doc_df)] = item

    if os.path.exists('data/index_to_doc_id.pkl'):
        with open('data/index_to_doc_id.pkl', 'rb') as f:
            doc_ids = pickle.load(f)


    out_df = pd.DataFrame(columns = ['claim_id', 'doc_id', 'prompt'])
    for index, row in tqdm(df.iterrows()):
        # get claim
        claim = claim_df[index]

        #get doc
        retrieved_doc_ids = row['doc_id']
        for doc_id in retrieved_doc_ids:
            doc_index = doc_ids.index(doc_id)
            paper = doc_df[doc_index]

            # form prompt
            prompt = form_prompt(paper, claim)
            out_df.loc[len(out_df)] = {'claim_id': claim['id'], 'doc_id': doc_id, 'prompt': prompt}

    out_df.to_csv('data/prompts.csv')