import numpy as np 
import json 
import pandas as pd 
import string 
import codecs 
from tqdm import tqdm
from nltk.corpus import stopwords
from multiprocessing.pool import ThreadPool as Pool 
import concurrent.futures

STOP_WORDS = stopwords.words('english')
STOP_WORDS.extend(['l', 'u', 'm'])    

def worker(item):
    try:
        return build(item)
    except:
        print('Error!')    

def build(a, b, c, d):
    a[b[c]] = [x for x,y in d.items() if y == b[c]]
    return a

def split_and_convert(line):
    return [x.lower() for x in line.split()]

def extract_corpus(file_path):
    with open(file_path, 'r') as file_handle:
        corpus = file_handle.readlines()
        print('Splitting into words and making lower case: ')
        split_corpus = []
        for i in tqdm(range(len(corpus))):
            split_corpus.append(split_and_convert(corpus[i]))
           
        print('Removing punctuations:')
        for i in tqdm(range(len(split_corpus))):
            split_corpus[i] = [x.translate(str.maketrans('','', string.punctuation+' 0123456789')) for x in split_corpus[i]]  

        print('Removing Stop words:')
        for i in tqdm(range(len(split_corpus))):
            split_corpus[i] = [x for x in split_corpus[i] if x not in STOP_WORDS and x != ""]

        print('Combining into a single dimension array of words:')
        words = [word for l in split_corpus for word in l]

        return dict(enumerate(words, 1)), list(set(words))

def build_inverted_index(words_enum, words_set):
    IV = dict()
    print('Building Inverted Index: ')
    
    for i in tqdm(range(len(words_set))):
        IV[words_set[i]] = [x for x,y in words_enum.items() if y == words_set[i]]
        #build((IV, words_set, i, words_enum))
    '''
    executor = concurrent.futures.ProcessPoolExecutor(10)
    futures = [executor.submit(worker, item) for item in [(IV, words_set, i, words_enum) for i in tqdm(range(len(words_set)))]]
    concurrent.futures.wait(futures)    
    
    
    pool_size = 10
    pool = Pool(pool_size)
    for i in tqdm(range(len(words_set))):
        async_result = pool.apply_async(build, (IV, words_set, i, words_enum))
        IV = async_result.get()
    '''   
    print('Inverted index construction completed')
    return IV        


if __name__ == '__main__':
    words_enum, words_set = extract_corpus('.\\corpus\\movie_lines.txt')
    
    inverted_index = build_inverted_index(words_enum, words_set)
    saved_inverted_index = {'doc0': inverted_index}


    with open('.\\inverted_index\\saved_inverted_index.json', 'w') as json_handle:
        json.dump(saved_inverted_index, json_handle)
        print('Inverted index saved as JSON')