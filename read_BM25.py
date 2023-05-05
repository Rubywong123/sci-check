import pandas as pd
import ast
import jsonlines

def parse_list(s):
    return ast.literal_eval(s) if isinstance(s, str) else s

if __name__ == '__main__':
    df = pd.DataFrame(columns = ['id', 'claim', 'evidence'])
    with open('data/claims.jsonl', 'r', encoding = 'utf-8') as f:
        for item in jsonlines.Reader(f):
            df.loc[len(df)] = item
    
    num_selected = 0
    tot = 0
    out = 0
    result_df = pd.read_csv('Oracle_input.csv', converters={'doc_id': parse_list})
    top_k = 5
    for index, row in result_df.iterrows():
        selected_papers = row['doc_id'][-top_k:]
        
        gold_row = df.iloc[index]
        if len(gold_row['evidence'].keys()) > 5:
            out += len(gold_row['evidence'].keys()) - 5
        tot += len(gold_row['evidence'].keys())
        for paper in selected_papers:
            if str(paper) in gold_row['evidence'].keys():
                num_selected += 1
                
    
    print(num_selected, tot, out)


    