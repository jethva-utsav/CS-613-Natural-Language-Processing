import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.tokenize import TweetTokenizer
from collections import Counter
from scipy.optimize import curve_fit 

#%%

tweets_df = pd.read_csv(r"C:\Users\DELL\Downloads\tweets-dataset.csv")
tweets_df = tweets_df.apply(lambda string: string.astype(str).str.lower())

tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

tweets_df['tokens_list'] = tweets_df['Sentence'].apply(lambda string: tknzr.tokenize(string))

#%%

tokens_list = [item for sublist in tweets_df['tokens_list'].tolist() for item in sublist]

print("total # of tokens:",len(tokens_list))
types_set = set(tokens_list)
print("total # of types:",len(types_set))
print("TTR:",len(types_set)/len(tokens_list))

#%%

words_list = ['teen','today','about','new','match','good','going']

words_df = pd.DataFrame({'words_list':words_list})

words_df['synonyms_list'] = words_df['words_list'].apply(lambda word: wordnet.synsets(word))

words_df['# of synonyms'] = words_df['synonyms_list'].apply(lambda synonyms_list: len(synonyms_list))

words_df['frequency'] = words_df['words_list'].apply(lambda word: Counter(tokens_list)[word])

words_df['word length'] = words_df['words_list'].apply(lambda word: len(word))

print(words_df['words_list'])
print(words_df['# of synonyms'])

plt.scatter(words_df['# of synonyms'],words_df['frequency'])
plt.plot(words_df['# of synonyms'],words_df['frequency'])
plt.xlabel('frequency')
plt.ylabel('# of synonyms')
plt.title("Zipf'Law")
plt.show()

tokens_frequency_list = Counter(tokens_list)

counts_df = pd.DataFrame.from_dict(tokens_frequency_list, orient='index')
counts_df.reset_index(level=0, inplace=True)
counts_df.columns = ['words', 'frequency']

counts_df['length of word']=counts_df['words'].apply(lambda word: len(word))

plt.plot(counts_df['length of word'],counts_df['frequency'])
plt.xlabel('frequency')
plt.ylabel('length of word')
plt.show()

#%%
# # Q3

N = []
V = []
for i in range(3500):
    total_tokens = tokens_list[:i*100]
    N.append(len(total_tokens))
    V.append(len(set(total_tokens)))

plt.plot(N,V,)
plt.xlabel('Tokens (N)')
plt.ylabel('Vocabulary |V|')
plt.title('Heaps law')
plt.show()

def heap(N, K, b):
    V = K*(N**b)
    return V

param, param_cov = curve_fit(heap, N, V) 

K = param[0]
b = param[1]

ans = K*(N**b) 

plt.plot(N, V, 'o', color ='pink', label ="original data") 
plt.plot(N, ans, '--', color ='blue', label ="curve fitted data") 
plt.xlabel('Tokens (N)')
plt.ylabel('Vocabulary |V|')
plt.title('Heaps law')
plt.legend() 
plt.show() 





    


