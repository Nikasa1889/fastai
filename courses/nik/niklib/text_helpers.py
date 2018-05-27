import html
from fastai.text import *

re1 = re.compile(r'  +')

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag
NUM = 'xdig'  # Number

def split_by_popularity(word, w2v):
    unknown_pop_score = w2v.n_vocabs
    if w2v.word2idx(word) > 0:
        return w2v.word2idx(word), [word]
        
    if (word.replace('.','',1).replace(',','',1).isdigit()): #all numbers replaced by xdig. except single digit
        # return unknown_pop_score, [word] # Don't care about digit
        return unknown_pop_score, [f'{NUM}']

    if len(word)<=5:
        return unknown_pop_score, [word]
    
    best_pop_score = unknown_pop_score
    best_split = None
    best_nsplit = 3 # Max split to split
    for i_cut in range(len(word)-2, 1, -1):
        prefix, core = word[:i_cut], word[i_cut:]

        if w2v.word2idx(core) < 0:
            continue
        core_score = w2v.word2idx(core)

        if w2v.word2idx(prefix) < 0:
            prefix_score, prefix = split_by_popularity(prefix, w2v)
        else:
            prefix_score, prefix = w2v.word2idx(prefix), [prefix]                
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


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ').replace('g.', 'g').replace('pr.kg', 'pr kg')
    return re1.sub(' ', html.unescape(x))

def get_texts(df, w2v, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'{BOS} ' + df[n_lbls].astype(str)
    texts = texts.apply(fixup).values.astype(str)

    alltoks = Tokenizer().proc_all_mp(partition_by_cores(texts)) # Note: apply our split_by_popularity algorithm
    alltoks = [[split_by_popularity(tok, w2v)[1] for tok in toks] for toks in alltoks]
    alltoks = [[newtok for newtoks in toks for newtok in newtoks] for toks in alltoks] # flatten list
    return alltoks, list(labels)
