'''
Goal: checking results of post-processed model predictions

Two settings:
1. FEVER: Retrieval + Judging
2. ORACLE: gold ECAP (+ negative samples) + Judging

Metrices:
1. accuracy of ECAPs (NEI not included) (1, 2)
2. accuracy of all predicted pairs (NEI included) (1, 2)
3. confusion matrix
'''
import pandas as pd
import jsonlines

enum = {"NEI": 0, "SUPPORT": 1, "CONTRADICT": 2}

def equal_rows(res_row, gold_row):
    return (res_row['claim_id'] == gold_row['id'] and res_row['doc_id'] == gold_row['doc_id'] and res_row['response'] == gold_row['label'])

def ECAP_acc(response_df: pd.DataFrame, gold_df: pd.DataFrame):
    correct_pred = 0

    lower_idx = 0
    upper_idx = 0
    for idx, row in response_df.iterrows():
        id = row['claim_id']
        while(upper_idx < gold_df.shape[0] and gold_df.iloc[upper_idx]['id'] == id):
            upper_idx += 1
        for i in range(lower_idx, upper_idx):
            if equal_rows(row, gold_df.iloc[i]):
                correct_pred += 1
                break
        
        lower_idx = upper_idx
        
    print('=============ECAP ACCURACY=============')
    print(correct_pred / gold_df.shape[0])
    return correct_pred / gold_df.shape[0]


def predict_acc(response_df: pd.DataFrame, gold_df: pd.DataFrame):

    correct_pred = 0

    lower_idx = 0
    upper_idx = 0
    for idx, row in response_df.iterrows():
        id = row['claim_id']
        while(upper_idx < gold_df.shape[0] and gold_df.iloc[upper_idx]['id'] == id):
            upper_idx += 1
        if id == gold_df.iloc[lower_idx]['id']:
            have_doc = False
            for i in range(lower_idx, upper_idx):
                if equal_rows(row, gold_df.iloc[i]):
                    have_doc = True
                    correct_pred += 1
                    break
                elif row['doc_id'] == gold_df.iloc[i]['doc_id']:
                    have_doc = True
                    break

            # check NEI
            if not have_doc and row['response'] == 0:
                correct_pred += 1
            
        else:
            # check if response is NEI
            if row['response'] == 0:
                correct_pred += 1
        lower_idx = upper_idx
    print('=============PREDICTION ACCURACY=============')
    print(correct_pred / response_df.shape[0])

    return correct_pred / response_df.shape[0]


def compute_F1(TP, FP, FN):
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    return precision, recall, f1
def Oracle_3_class_F1(response_df: pd.DataFrame, gold_df: pd.DataFrame):
    S_TP, S_FP, S_FN = 0, 0, 0
    C_TP, C_FP, C_FN = 0, 0, 0
    N_TP, N_FP, N_FN = 0, 0, 0

    lower_idx = 0
    upper_idx = 0
    for idx, row in response_df.iterrows():
        id = row['claim_id']
        while(upper_idx < gold_df.shape[0] and gold_df.iloc[upper_idx]['id'] == id):
            upper_idx += 1
        if id == gold_df.iloc[lower_idx]['id']:
            have_doc = False
            for i in range(lower_idx, upper_idx):
                if equal_rows(row, gold_df.iloc[i]):
                    have_doc = True
                    
                    if row['response'] == 1:
                        S_TP += 1
                    elif row['response'] == 2:
                        C_TP += 1

                    break

                elif row['doc_id'] == gold_df.iloc[i]['doc_id']:
                    if gold_df.iloc[i]['label'] == 1:
                        S_FN += 1
                        if row['response'] == 2:
                            C_FP += 1
                        if row['response'] == 0:
                            N_FP += 1

                    elif gold_df.iloc[i]['label'] == 2:
                        C_FN += 1
                        if row['response'] == 1:
                            S_FP += 1
                        if row['response'] == 0:
                            N_FP += 1
                    

                    have_doc = True
                    break

            # check NEI
            if not have_doc:
                if row['response'] == 0:
                    N_TP += 1
                elif row['response'] == 1:
                    S_FP += 1
                elif row['response'] == 2:
                    C_FP += 1
            
        else:
            # check if response is NEI
            if row['response'] == 0:
                N_TP += 1
            elif row['response'] == 1:
                S_FP += 1
            elif row['response'] == 2:
                C_FP += 1

        lower_idx = upper_idx
    
    S_precision, S_recall, S_f1 = compute_F1(S_TP, S_FP, S_FN)
    C_precision, C_recall, C_f1 = compute_F1(C_TP, C_FP, C_FN)
    N_precision, N_recall, N_f1 = compute_F1(N_TP, N_FP, N_FN)
    micro_precision, micro_recall, micro_f1 = compute_F1(S_TP + C_TP + N_TP, S_FP+C_FP+N_FP, S_FN+C_FN+N_FN)

    print('=========SUPPORT RESULT==============')
    print('precision = {}, recall = {}, f1 = {}'.format(S_precision, S_recall, S_f1))
    print('=========CONTRADICT RESULT==============')
    print('precision = {}, recall = {}, f1 = {}'.format(C_precision, C_recall, C_f1))
    print('=========NOINFO RESULT==============')
    print('precision = {}, recall = {}, f1 = {}'.format(N_precision, N_recall, N_f1))
    print('=========MICRO RESULT==============')
    print('precision = {}, recall = {}, f1 = {}'.format(micro_precision, micro_recall, micro_f1))
    return micro_precision, micro_recall, micro_f1

def FEVER_2_class_F1(response_df: pd.DataFrame, gold_df: pd.DataFrame):
    S_TP, S_FP, S_FN = 0, 0, 0
    C_TP, C_FP, C_FN = 0, 0, 0

    lower_idx = 0
    upper_idx = 0
    for idx, row in response_df.iterrows():
        id = row['claim_id']
        while(upper_idx < gold_df.shape[0] and gold_df.iloc[upper_idx]['id'] == id):
            upper_idx += 1
        if id == gold_df.iloc[lower_idx]['id']:
            have_doc = False
            for i in range(lower_idx, upper_idx):
                if equal_rows(row, gold_df.iloc[i]):
                    have_doc = True
                    
                    if row['response'] == 1:
                        S_TP += 1
                    elif row['response'] == 2:
                        C_TP += 1

                    break
                elif row['doc_id'] == gold_df.iloc[i]['doc_id']:
                    if gold_df.iloc[i]['label'] == 1:
                        S_FN += 1
                        if row['response'] == 2:
                            C_FP += 1
                    
                    elif gold_df.iloc[i]['label'] == 2:
                        C_FN += 1
                        if row['response'] == 1:
                            S_FP += 1

                    have_doc = True
                    break

            # check NEI
            if not have_doc and row['response'] != 0:
                if row['response'] == 1:
                    S_FP += 1
                elif row['response'] == 2:
                    C_FP += 1
            
        else:
            # check if response is NEI
            if row['response'] != 0:
                if row['response'] == 1:
                    S_FP += 1
                elif row['response'] == 2:
                    C_FP += 1

        lower_idx = upper_idx
    
    S_precision, S_recall, S_f1 = compute_F1(S_TP, S_FP, S_FN)
    C_precision, C_recall, C_f1 = compute_F1(C_TP, C_FP, C_FN)
    micro_precision, micro_recall, micro_f1 = compute_F1(S_TP + C_TP, S_FP+C_FP, S_FN+C_FN)

    print('=========SUPPORT RESULT==============')
    print('precision = {}, recall = {}, f1 = {}'.format(S_precision, S_recall, S_f1))
    print('=========CONTRADICT RESULT==============')
    print('precision = {}, recall = {}, f1 = {}'.format(C_precision, C_recall, C_f1))
    print('=========MICRO RESULT==============')
    print('precision = {}, recall = {}, f1 = {}'.format(micro_precision, micro_recall, micro_f1))
    return micro_precision, micro_recall, micro_f1
    

if __name__ == '__main__':
    model = 'gpt-3.5-turbo'
    response_df = pd.read_csv('data/{}_parsed.csv'.format(model))

    # read claim into df
    df = pd.DataFrame(columns = ['id', 'claim', 'evidence'])
    with open('data/claims.jsonl', 'r', encoding = 'utf-8') as f:
        for item in jsonlines.Reader(f):
            df.loc[len(df)] = item
        
        
    flatten_df = pd.DataFrame(columns= ['id', 'claim', 'doc_id', 'provenance', 'label', 'sentences', 'model_ranks'])

    # NEI not included
    for row in df.index:
        for key in df.loc[row]['evidence']:
            flatten_df.loc[len(flatten_df)] = {'id': df.loc[row]['id'],'claim': df.loc[row]['claim'], 'doc_id': int(key), 'provenance': df.loc[row]['evidence'][key]['provenance'], 
            'label': enum[df.loc[row]['evidence'][key]['label']], 'sentences': df.loc[row]['evidence'][key]['sentences'], 'model_ranks': df.loc[row]['evidence'][key]['model_ranks']}
        
    # ECAP acc
    pred_ecap_acc = ECAP_acc(response_df, flatten_df)
    pred_acc = predict_acc(response_df, flatten_df)
    pref_Oracle_f1 = Oracle_3_class_F1(response_df, flatten_df)
    pred_fever_f1 = FEVER_2_class_F1(response_df, flatten_df)

