import jsonlines
from tqdm import tqdm
import pandas as pd


if __name__ == '__main__':

    df = pd.DataFrame(columns = ['id', 'claim', 'evidence'])
    with open('data/claims.jsonl', 'r', encoding = 'utf-8') as f:
        for item in jsonlines.Reader(f):
            df.loc[len(df)] = item
    
    out_df = pd.DataFrame(columns = ['id', 'doc_id'])

    for index, row in tqdm(df.iterrows()):
        selected_papers = map(int, list(row['evidence'].keys()))
        print(selected_papers)

        out_df.loc[len(out_df)] = {'id': row['id'], 'doc_id': selected_papers}
        

    out_df.to_csv('Oracle_input.csv')

