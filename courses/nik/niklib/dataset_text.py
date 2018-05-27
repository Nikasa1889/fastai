import sys
sys.path.append("..")
from falib.core import V, T, to_np
from falib.dataset import BaseDataset, ModelData
from falib.dataloader import DataLoader
import numpy as np
import pandas as pd
import dill as pickle
from sklearn.model_selection import train_test_split

class TextDataset(BaseDataset):
    # Convert text and labels in string to id so that it can be represented as numpy array
    # Each word is converted to its id using the word2vec id
    # Each label is converted to a unique id, which can be convert back using id2label function
    def __init__(self, xs, ys, idx2label):
        self._xs = xs
        self._ys = ys
        self._idx2label = idx2label
        self._label2idx = {self._idx2label[idx]: idx for idx in range(len(self._idx2label))}
        super().__init__()
    
    def extend(self, xs, ys):
        self._xs = np.append(self._xs, xs, axis=0)
        self._ys = np.append(self._ys, ys, axis=0)
        
    def save(self, fname):
        pickle.dump( (self.xs, self.ys, self._idx2label), open( fname, "wb" ))
    
    def idx2label(self, idx):
        return self._idx2label[idx]
    
    def label2idx(self, label):
        return self._label2idx.get(label, None)
    
    def label_dist(self):
        count = {}
        for val in self.ys:
            label = self.idx2label(val)
            count[label] = count.get(label, 0) + 1
        return count
    
    def summary(self):
        print(self)
       
    def __repr__(self):
        summary  = f'Number of examples: {self.get_n()} \n'
        summary += f'Number of classes: {self.get_c()} \n'
        summary += f'Fixed sequence length: {self.get_seqlen()} \n'
        summary += f'Size of xs: {self.get_sz()} \n'
        summary += f'Is regression task: {self.is_reg()}'
        return summary
    
    @classmethod
    def from_textlabels(cls, texts, labels, text_processor):
        idx2label = list(set(labels))
        label2idx = {idx2label[idx]: idx for idx in range(len(idx2label))}
        
        ys = [label2idx[label] for label in labels]
        ys = np.asarray(ys)
        xs = text_processor.text2features(texts)
        return cls(xs, ys, idx2label)
    
    @classmethod
    def from_mattext(cls, mattext, text_processor):
        df = mattext.df
        df = df.assign(text = lambda df: df.prodName + " " + df.prodDesc)[["catName", "text"]]
        df = df.rename(columns = {"catName": "label", "text":"text"})
        return cls.from_dataframe(df, text_processor)
    
    @classmethod
    def from_dataframe(cls, df, text_processor):
        return cls.from_textlabels(df.text, df.label, text_processor)
    
    @classmethod
    def from_pickle(cls, fname):
        xs, ys, idx2label = pickle.load( open( fname, "rb" ) )
        return cls(xs, ys, idx2label)
    
    @classmethod
    def from_csv(cls, fname, text_processor):
        df = pd.read_csv(fname)
        return cls.from_dataframe(df, text_processor)
    
    def subset(self, ids):
        return self.__class__(self._xs[ids], self._ys[ids], self._idx2label)
    
    def __len__(self): return self.get_n()

    def get_seqlen(self): return self._xs.shape[1]
    
    def get_n(self): return len(self._ys)
    
    def get_c(self): return int(self._ys.max()) + 1
    
    def get_sz(self): return self._xs.shape

    def get_x(self, i): return self._xs[i]

    def get_y(self, i): return self._ys[i]
    
    @property
    def ys(self): return self._ys
    
    @property
    def xs(self): return self._xs
    
    def is_multi(self): return False

    def is_reg(self): return False

    
class TextModelData(ModelData):
    def __init__(self, path, trn_ds, val_ds, test_ds, bs, nworkers):
        """
            trn_ds: train dataset
            val_ds: validation dataset
            test_ds: test dataset, can be None
            bs: batchsize
        """
        self.path, self.bs, self.nworkers = path, bs, nworkers
        self.trn_dl, self.val_dl = self.get_dl(trn_ds, True), self.get_dl(val_ds, False)
        self.test_dl = None if test_ds is None else self.get_dl(test_ds, False)
        
    def summary(self):
        print(self)
    
    def __repr__(self):
        summary  = f'Data path: {self.path} \n'
        summary += f'Fixed sequence length: {self.get_seqlen()} \n'
        summary += f'Number of classes: {self.get_c()} \n'
        summary += f'Training examples: {len(self.trn_ds)} \n'
        summary += f'Validation examples: {len(self.val_ds)} \n'
        test_len = 0 if self.test_dl is None else len(self.test_ds)
        summary += f'Test examples: {test_len} \n'
        return summary
    
    def get_dl(self, ds, shuffle):
        """
            Return a dataloader for a dataset
            
            ds: dataset
            shuffle: shuffle or not
        """
        return DataLoader(ds, batch_size=self.bs, shuffle=shuffle,
                              num_workers=self.nworkers, drop_last=True, pin_memory=False)

    @classmethod
    def from_ds(cls, path, ds, val_p=0.2, test_p=0.0, bs=128, nworkers=1, stratify=True, random_seed=42):
        df = pd.DataFrame({'y': ds.ys})
        strat_series = df.y if stratify else None
        train_df, val_df = train_test_split(df, test_size=val_p, stratify=strat_series, random_state=random_seed)
        if (test_p > 0):
            strat_series = train_df.y if stratify else None
            train_df, test_df = train_test_split(train_df, test_size=test_p, stratify=strat_series, random_state=random_seed)
        
        trn_ds, val_ds = ds.subset(train_df.index), ds.subset(val_df.index)
        test_ds = ds.subset(test_df.index) if test_p > 0 else None
        return cls(path,trn_ds, val_ds, test_ds, bs, nworkers)
    
    def idx2label(self, idx):
        return self.trn_ds.idx2label(idx)
    
    def label2idx(self, label):
        return self.trn_ds.label2idx(label)
    
    def get_seqlen(self): return self.trn_ds.get_seqlen()
    
    def get_c(self): return self.trn_ds.get_c()
    @property
    def is_reg(self): return self.trn_ds.is_reg
    @property
    def is_multi(self): return self.trn_ds.is_multi
    @property
    def trn_ds(self): return self.trn_dl.dataset
    @property
    def val_ds(self): return self.val_dl.dataset
    @property
    def test_ds(self): return self.test_dl.dataset
    @property
    def trn_y(self): return self.trn_ds.ys
    @property
    def val_y(self): return self.val_ds.ys
