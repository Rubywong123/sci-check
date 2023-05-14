'''

Post-process
Goal: Parsing text outputs into labels that can be automatically tested
'''
import pandas as pd
import argparse
enum = {"Neither": 0, "True": 1, "False": 2}

def post_process(response: str):
    # Naive parsing: splitting with '.' & fetch the first part
    ans = response.split('.')[0].split()[0]
    return enum[ans] if ('Neither' in ans or 'True' in ans or 'False' in ans) else -1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='choose model to prompt', choices=['gpt-3.5-turbo', 'galactica-mini', 'galactica-base',
                                                                           'galactica-standard', 'galactica-large', 'galactica-huge']
                                                                , default='galactica-mini')
    args = parser.parse_args()
    response_df = pd.read_csv('data/{}_out.csv'.format(args.model))

    # print(response_df, response_df.columns)
    response_df = response_df.drop('Unnamed: 0', axis=1)
    response_df['response'] = response_df['response'].apply(post_process)

    response_df.to_csv('data/{}_parsed.csv'.format(args.model))
