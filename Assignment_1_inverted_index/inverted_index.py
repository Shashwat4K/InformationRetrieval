from collections import Counter
import string
import time
import json
import os
import sys
import math
from tqdm import tqdm
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from nltk.corpus import stopwords

STOP_WORDS = stopwords.words('english')
STOP_WORDS.extend(['l', 'u', 'm'])

def binary_search(arr, low, high, x):
    
    while low <= high:
        mid = int(low + (high - low)/2)
        
        if arr[mid] == x:
            return mid 
        elif arr[mid] > x:
            high = mid-1 
        else:
            low = mid +1
    
    return high                         

def compute_time(IV, params):
    start_time = time.time()
    _ = IV.next_phrase(params[0], params[1], params[2])
    end_time = time.time()
    #print('Outcome = {} Time taken: {}'.format(_, end_time-start_time))
    return (end_time-start_time)

def get_phrase_list(file_path):
    with open(file_path, 'r') as fp:
        clean_list = []
        phrases = fp.readlines()
        phrases = [x.lower().translate(str.maketrans('', '', string.punctuation+'0123456789—’“…”')) for x in phrases]
        phrases = [x.split() for x in phrases]
        for i in phrases:
            clean_list.append([x for x in i if x not in STOP_WORDS and x != ""])

        return clean_list


class InvertedIndex(object):
    
    def __init__(self, corpus_path=None, docID=0, already_present=False, saved_dict=None):
        if already_present==False:
            self.inverted_index = {}
            self.docID = None
            self.docID = docID

            def remove_punctuations(corpus_list):
                punctuations_removed = [corpus_list[i].translate(str.maketrans('', '', string.punctuation+'0123456789—’“…”')) for i in range(len(corpus_list))]
                return punctuations_removed

            start_time = time.time() 
            corpus = open(corpus_path, 'r')
            corpus_list = corpus.readlines()
            corpus_list = remove_punctuations(corpus_list)
            corpus_sentences = [
                l.strip() for l in list(
                    filter(
                        lambda a: a != '\n', [line for i in [line.split('.') for line in corpus_list] for line in i]
                    )
                )
            ]
            self.words = [word for word in [line.split() for line in corpus_sentences]]
            self.words = [word.lower() for l in self.words for word in l]
            self.word_count = Counter(self.words)
            self.words = dict(enumerate(self.words, 1))
            self.build_inverted_index()
            end_time = time.time()
            print('Time taken to build INVERTED INDEX for docID {} = {}'.format(self.docID, end_time-start_time))
            corpus.close()
        else:
            self.inverted_index = saved_dict
            self.docID = 0    

    def __eq__(self, other):
        return True if self.docID == other.get_docID() else False

    def __ne__(self, other):
        return True if self.docID != other.get_docID() else False

    def get_docID(self):
        return self.docID

    def get_words(self):
        return self.word_count

    def total_words(self):
        s = 0
        for _,i in self.word_count.items():
            s += i
        return s    

    def build_inverted_index(self):
        for word, _ in self.word_count.items():
            self.inverted_index[word] = sorted([i for i, j in self.words.items() if j == word])
    
    def print_inverted_index(self):
        for i, j in self.inverted_index.items():
            print('{} -> {}'.format(i, j))

    def save_inverted_index(self, file_path):
        with open(file_path, 'w') as file_handle:
            json.dump(self.inverted_index, file_handle)

    def get_longest_posting_list(self):
        positing_lists = self.inverted_index.items()
        max_len = 0
        freq_word = None
        for i,j in positing_lists:
            if len(j) > max_len:
                max_len = len(i)
                freq_word = i
        return freq_word, max_len      

    def get_next_binary(self, word, position):
        if word not in self.inverted_index.keys():
            return -1
        Pt = self.inverted_index[word]
        lt = len(Pt)
        if lt == 0 or Pt[lt-1] <= position:
            return sys.maxsize
        if Pt[0] > position:
            return Pt[0]
   
        i = binary_search(Pt, 0, lt-1, position)
        return Pt[i+1] # i is position of 'position', so i+1 will be required position 

    def get_next_linear(self, word, position):
        if word not in self.inverted_index.keys():
            return -1 
        Pt = self.inverted_index[word]
        lt = len(Pt) 
        if lt == 0 or Pt[lt-1] <= position:
            return sys.maxsize 
        if Pt[0] > position:
            return Pt[0]
        i = 0    
        while Pt[i] <= position:
            i += 1
        return Pt[i]           

    def get_next_galloping(self, word, position):
        if word not in self.inverted_index.keys():
            return -1 
        Pt = self.inverted_index[word]
        lt = len(Pt)
        if lt == 0 or Pt[lt-1] <= position:
            return sys.maxsize
        if Pt[0] >  position:
            return Pt[0]
        low = 0
        jump = 1
        high = low + jump 
        while high < lt and Pt[high] <= position:
            low = high 
            jump = 2*jump 
            high = low + jump 
        if high > lt-1:
            high = lt-1 
        i = binary_search(Pt, low, high, position)
        return Pt[i+1]                 

    def get_previous(self, word, position):
        pass

    def next_phrase(self, phrase, position, function_number):
        
        if function_number == 0:
            next_function = self.get_next_linear
        elif function_number == 1:
            next_function = self.get_next_binary
        else:
            next_function = self.get_next_galloping       
        
        u = next_function(phrase[0], position)
        if u == -1:
            return (sys.maxsize, sys.maxsize)
        v = u 
        for i in range(1, len(phrase)):
            v = next_function(phrase[i], v)
            if v == -1:
                return (sys.maxsize, sys.maxsize)
        if v == sys.maxsize:
            return (sys.maxsize, sys.maxsize)
        if v-u == len(phrase)-1:
            return (u, v)
        else:
            return self.next_phrase(phrase, v-len(phrase), function_number)     

def select_random_phrases(corpus_path, number=1000):
    with open(corpus_path, 'r') as fp:
        corpus = np.array(fp.readlines())
        random_indices = np.random.randint(0, len(corpus)-1, number)
        phrases = corpus[random_indices]
        with open('.\\phrases.txt', 'w') as phrase_writer:
            phrase_writer.writelines(list(phrases))

def sort_according_to_lengths(phrase_list):
    return sorted(phrase_list, key=len)

def print_phrases(phrase_list):
    for i in phrase_list:
        print('{} -> {}'.format(i, len(i)))



if __name__ == "__main__":
    
    print('Loading saved index...')
    df = pd.read_json('.\\inverted_index\\saved_inverted_index.json')
    print('Inverted index loaded')
    ii = dict(df['doc0'])
    ii = InvertedIndex(already_present=True, saved_dict=ii)

    # Selection of 1000 random phrases
    print('Selecting random phrases:')
    number = 2000
    select_random_phrases('.\\corpus\\movie_lines.txt', number)
    phrases = get_phrase_list('.\\phrases_backup.txt')
    phrases_copy = phrases
    # print_phrases(phrases[:50])
    phrases = sort_according_to_lengths(phrases)
    
    length_vector = [len(x) for x in phrases]

    
    # Time computation
    print('Computing times: ')
    linear_time = []
    binary_time = []
    galloping_time = []

    for i in tqdm(range(len(phrases))):
        linear_time.append(compute_time(ii, (phrases[i], 0, 0)))
        binary_time.append(compute_time(ii, (phrases[i], 0, 1)))
        galloping_time.append(compute_time(ii, (phrases[i], 0, 2)))

    # Plotting time taken for each phrase versus the phrase number

    print('\nPlotting the time taken by \'nextPhrase\' each phrase with different methods versus the phrase number')
    plt.plot(range(1, number+1), linear_time, label='linear next', color='r')
    plt.plot(range(1, number+1), binary_time, label='binary next', color='g')
    plt.plot(range(1, number+1), galloping_time, label='galloping next', color='b')
    plt.xlabel('Phrase number')
    plt.ylabel('Response Times')
    plt.title('Time comparison')
    plt.legend(loc='upper right')
    plt.show()

    # Calculating Average response times 
    mean_linear = np.mean(linear_time)
    mean_binary = np.mean(binary_time)
    mean_galloping = np.mean(galloping_time)
    print('\nAverage Response times: ')
    print('Next Phrase using Linear scan: {}\nNext Phrase using Binary search: {}\nNext Phrase using Galloping Search: {}'.format(mean_linear, mean_binary, mean_galloping))
    
    # Plotting response times for different lengths of phrases
    print('\nPlotting response times versus length of phrases')
    plt.plot(linear_time, length_vector,'ro', label='L')
    plt.plot(binary_time, length_vector,'go', label='B')
    plt.plot(galloping_time, length_vector,'bo', label='G')
    plt.xlabel('Response times')
    plt.ylabel('Length of phrase')
    plt.legend(loc='upper right')
    plt.title('Response times versus Length of Phrases')
    plt.show()
    
    # phrases of length 2
    phrases_2 = get_phrase_list('.\\phrases_length2.txt')
    #print(phrases_2)
    l = []
    b = []
    g = []
    max_len, word = ii.get_longest_posting_list()
    for i in tqdm(range(len(phrases_2))):
        l.append(compute_time(ii, (phrases_2[i], 0, 0)))
        b.append(compute_time(ii, (phrases_2[i], 0, 1)))
        g.append(compute_time(ii, (phrases_2[i], 0, 2)))

    plt.plot(range(1, 21), l, label='linear next', color='r')
    plt.plot(range(1, 21), b, label='binary next', color='g')
    plt.plot(range(1, 21), g, label='galloping next', color='b')
    plt.xlabel('Phrase number')
    plt.ylabel('Response Times')
    plt.title('Time comparison')
    plt.legend(loc='upper right')
    plt.show()