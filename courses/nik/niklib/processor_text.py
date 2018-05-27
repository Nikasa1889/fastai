from torchtext import vocab, data
import numpy as np
from niklib.utils import pad_sequences
#from torchtext.datasets import language_modeling
#from spacy import spacy
# Download spacy package if does not have
#!python -m spacy download en
import string
import re
from falib.text import Tokenizer

class TextProcessor(object):
    
    def __init__(self, w2v, tokenize_fn=None, max_pad=50):
        self.w2v, self.tokenize_fn, self.max_pad = w2v, tokenize_fn, max_pad
        self.re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
        self.fastai_tok = Tokenizer()
        if tokenize_fn is None:
            TEXT = data.Field(lower=True, tokenize=self.tokenize)
            self.tokenize_fn = TEXT.preprocess

    def text2features(self, texts):
        if isinstance(texts, str): texts = [texts]
        tk_texts = [self.tokenize_fn(txt) for txt in texts]
        lens = np.array([len(txt) for txt in tk_texts])
        minl, maxl, meanl = lens.min(), lens.max(), lens.mean()
        print(f'Text length: min={minl}, max={maxl}, mean={meanl}')
        print(f'Padding all text to have fixed length = {self.max_pad}')
        
        xs = [[self.w2v.word2idx(word) for word in txt if self.w2v.word2idx(word) > 0] for txt in tk_texts]
        xs = pad_sequences(xs, maxlen=self.max_pad, padding='pre', truncating='post')
        xs = np.asarray(xs)
        return xs
    
    def __repr__(self):
        summary  = f'Tokenize function: {self.tokenize_fn.__name__} \n' 
        summary += f'Max zero-padding len: {self.max_pad} \n'
        return summary
    
    def split_by_punctuation(self, s): return self.re_tok.sub(r' ', s).split()

    def split_by_popularity(self, word):
        unknown_pop_score = self.w2v.n_vocabs
        if self.w2v.word2idx(word) > 0:
            return self.w2v.word2idx(word), [word]

        if len(word)<=5:
            return unknown_pop_score, [word]

        if (word.replace('.','',1).replace(',','',1).isdigit()):
            return unknown_pop_score, [word] # Don't care about digit

        best_pop_score = unknown_pop_score
        best_split = None
        best_nsplit = 3 # Max split to split
        for i_cut in range(len(word)-2, 1, -1):
            prefix, core = word[:i_cut], word[i_cut:]

            if self.w2v.word2idx(core) < 0:
                continue
            core_score = self.w2v.word2idx(core)

            if self.w2v.word2idx(prefix) < 0:
                prefix_score, prefix = self.split_by_popularity(prefix)
            else:
                prefix_score, prefix = self.w2v.word2idx(prefix), [prefix]                
            if (prefix_score >= unknown_pop_score): # Don't split if all splitted words are good
                continue
            pop_score = prefix_score + core_score
            words = prefix; words.append(core)

            if len(words) > best_nsplit: continue # Don't split more than best_nsplit
            if (len(words) < best_nsplit) or (pop_score < best_pop_score):
                #print(prefix, prefix_score, core_score, pop_score)
                best_split = words
                best_nsplit= len(words)
                best_pop_score = pop_score

        if best_split is None:
            return (unknown_pop_score, [word])
        else:
            return (best_pop_score, best_split)

    # TODO: Add new word to w2v with random vector if can't split well
    # TODO: Remove unpopular words from w2v, add special token for numbers or irrelevant words
   
    def tokenize(self, text):
        t = self.fastai_tok.spacy_tok(text)
        new_t = []
        for word in t:
            new_t.extend(self.split_by_punctuation(word))
        t = new_t
        new_t = []
        for word in t:
            word = word.lower()
            pop_score, words = self.split_by_popularity(word)
            new_t.extend(words)
            #if self.w2v.word2idx(word) < 0:
            #    print(words)
        return new_t

