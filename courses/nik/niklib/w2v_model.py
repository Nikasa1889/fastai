import numpy as np
from tqdm import tqdm
import _pickle as pickle
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors

class Word2VecModel:
    # word index must start from 1
    # word with idx=0 is empty word, which always has zero embedding
    def __init__(self, word2idx, idx2word, idx2emb):
        self._word2idx, self.idx2word, self.idx2emb = word2idx, idx2word, idx2emb
        if not all(v == 0 for v in self.idx2emb[0]):
            raise Exception('Embedding at idx = 0 must be all 0')
        self._neigh = None
    
    def summary(self):
        print(self)
    
    def __repr__(self):
        summary  = f'Number of vocabularies: {self.n_vocabs} \n'
        summary += f'Number of embedding factors: {self.n_embfactors} \n'
        return summary
    
    @property
    def n_embfactors(self):
        return self.idx2emb.shape[1]
    
    @property
    def n_vocabs(self):
        return self.idx2emb.shape[0]
    
    @property
    def emb_mat(self):
        return self.idx2emb
    
    @classmethod
    def from_fasttext_vec(cls, fname):
        with open(fname, encoding="utf-8") as f:
            nwords, nfeatures = [int(num) for num in f.readline().split()]
            nwords = nwords + 1 #Add empty word at idx 0
            idx2emb = np.zeros((nwords, nfeatures))
            idx2word = [None]*nwords
            word2idx = {}
            for i in tqdm(range(1, nwords)):
                try:
                    line = f.readline().strip()
                    word, emb = line.split(' ',maxsplit=1)
                    emb_vec = [float(num) for num in emb.split()]
                    if len(emb_vec) != nfeatures:
                        print(f'Error while reading word {i}, parsed word: {word} ')
                        continue
                    word2idx[word] = i
                    idx2word[i] = word
                    idx2emb[i,:] = emb_vec
                except Exception as ex:
                    print(ex)
        return cls(word2idx, idx2word, idx2emb)
    
    @classmethod
    def from_pickle(cls, fname):
        word2idx, idx2word, idx2emb = pickle.load( open( fname, "rb" ) )
        return cls(word2idx, idx2word, idx2emb)
    
    def save(self, fname):
        pickle.dump( (self.word2idx, self.idx2word, self.idx2emb), open( fname, "wb" ))

    #Maybe improve by splitting unknown word to 2 known words
    def word2emb(self, word):
        if word in self._word2idx:
            return self.idx2emb[self._word2idx[word]]
        else:
            raise NotImplemented
    
    def word2idx(self, word):
        if word in self._word2idx:
            return self._word2idx[word]
        else:
            return -1
    
    def idx2word(self, idx):
        return self.idx2word(idx)
    
    def dist(self, word1, word2):
        emb1 = self.word2emb(word1)
        emb2 = self.word2emb(word2)
        return self.emb_dist(emb1, emb2)
    
    def emb_dist(self, emb1, emb2):
        return cosine(emb1, emb2)
    
    def neighbors(self, word, k=10):
        if self._neigh is None:
            self._neigh = NearestNeighbors(n_neighbors=k, radius=0.5, metric='cosine', algorithm='brute')
            self._neigh.fit(self._idx2emb)
        distances, indices = self._neigh.kneighbors([self.word2emb(word)], n_neighbors=k)
        return [(self.idx2word[int(ind)], dist) for ind, dist in zip(list(indices[0]), list(distances[0]))]