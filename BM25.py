import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import scipy.sparse as sparse
import nltk
from nltk.stem import SnowballStemmer
import jsonlines
from tqdm import tqdm
import pandas as pd

def preprocess_corpus(documents, doc_ids):
    if os.path.exists('data/corpus_vocab_matrix.npy') and os.path.exists('data/corpus_doc_onehot.npy') and os.path.exists('data/corpus_idf.npy'):
        return
    print("============START PREPROCESSING=====================")
    print("==============FILTERING========================")
    # preprocessing sentences
    try:
        stop_words = stopwords.words('english')
    except:
        nltk.download('stopwords')
        stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    doc_tokens = [word_tokenize(doc.lower()) for doc in documents]
    doc_tokens = [[stemmer.stem(token) for token in doc if token not in stop_words] for doc in doc_tokens]
    
    print("===============ENCODING=========================")
    vectorizer = CountVectorizer()
    # sparse matrix
    term_freq = vectorizer.fit_transform([' '.join(doc) for doc in doc_tokens])
    vocab = vectorizer.get_feature_names_out()
    
    print("GOT vocabulary with length = ", len(vocab))

    # Compute the IDF values
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(term_freq)
    idf = transformer.idf_

    print("========IDF computation complete=================")

    # saving results
    with open('data/corpus_vocab_matrix.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open('data/corpus_doc_onehot.pkl', 'wb') as f:
        pickle.dump(term_freq, f)
    with open('data/corpus_idf.pkl', 'wb') as f:
        pickle.dump(idf, f)
    with open('data/index_to_doc_id.pkl', 'wb') as f:
        pickle.dump(doc_ids.to_list(), f)


def bm25(query, top_k = 5, k1=1.5, b=0.75):
    # Tokenize the query and documents
    query_tokens = word_tokenize(query.lower())
    

    # Remove stop words and stem the tokens
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    query_tokens = [stemmer.stem(token) for token in query_tokens if token not in stop_words]


    # Compute the document lengths
    doc_len = np.sum(term_freq, axis=1)

    # Compute the average document length
    avg_doc_len = np.mean(doc_len)

    # Compute the BM25 scores for each document
    scores = []
    for i, doc in tqdm(enumerate(term_freq)):
        score = 0
        for token in query_tokens:
            if token in vocab:
                # Compute the term frequency
                tf = term_freq[i, vocab.index(token)]
                # Compute the BM25 score for the current token
                score += idf[vocab.index(token)] * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len[i] / avg_doc_len))))
        scores.append(score[0, 0])

    # Return the BM25 scores
    return np.argsort(scores)[-top_k:]


df = pd.DataFrame(columns = ['doc_id', 'title', 'abstract', 'metadata'])
with open('data/corpus_candidates.jsonl', 'r', encoding = 'utf-8') as f:
    for item in jsonlines.Reader(f):
        df.loc[len(df)] = item
def list_to_str(l):
    return ' '.join(l)
df['abstract'] = df['abstract'].apply(list_to_str)

documents = (df['title'] + df['abstract']).to_list()

print("Number of papers: ", len(documents))

preprocess_corpus(documents, df['doc_id'])

if os.path.exists('data/corpus_vocab_matrix.pkl'):
    with open('data/corpus_vocab_matrix.pkl', 'rb') as f:
        vocab = pickle.load(f).tolist()
if os.path.exists('data/corpus_doc_onehot.pkl'):
    with open('data/corpus_doc_onehot.pkl', 'rb') as f:
        term_freq = pickle.load(f)
if os.path.exists('data/corpus_idf.pkl'):
    with open('data/corpus_idf.pkl', 'rb') as f:
        idf = pickle.load(f)
if os.path.exists('data/index_to_doc_id.pkl'):
    with open('data/index_to_doc_id.pkl', 'rb') as f:
        doc_ids = pickle.load(f)

df = pd.DataFrame(columns = ['id', 'claim', 'evidence'])
with open('data/claims.jsonl', 'r', encoding = 'utf-8') as f:
    for item in jsonlines.Reader(f):
        df.loc[len(df)] = item

num_selected = 0
for index, row in tqdm(df.iterrows()):
    selected_papers = bm25(row['claim'])
    print(f"Claim {row['id']}, selected papers: ", selected_papers)

    for p in selected_papers:
        if str(doc_ids[p]) in row['evidence'].keys():
            num_selected += 1

print(f"BM25 have sampled {num_selected} of {len(documents)}.")
