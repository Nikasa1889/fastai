from torch import nn
from falib.core import V, T, F
from falib.rnn_reg import EmbeddingDropout, WeightDrop, LockedDropout
import torch
# TODO: Muticlass Text Classification, Kaggle Competition, Sanfrancisco Crime Classification
# https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn
# TODO: Compare with the blog post for naives models
# http://nadbordrozd.github.io/blog/2017/08/12/looking-for-the-text-top-model/


# Check here for fastai Language Model and classifier
# https://github.com/fastai/fastai/blob/14902fb7fbf797c7a7307309aa74d96855a0fefd/fastai/lm_rnn.py#L23
# https://github.com/fastai/fastai/blob/14902fb7fbf797c7a7307309aa74d96855a0fefd/fastai/lm_rnn.py#L225
class WordEmbedding(nn.Module):
    def __init__(self, w2v, seqlen, drop_emb=0.02, drop_in=0.05):
        super().__init__()
        self.n_words, self.drop_emb = w2v.n_vocabs, drop_emb
        self.emb = nn.Embedding(w2v.n_vocabs, w2v.n_embfactors)
        self.emb.weight.data = torch.FloatTensor(w2v.emb_mat)
        self.emb.weight.requires_grad = True
        
        # self.emb_bn = nn.BatchNorm1d(seqlen) (Normal embeding and dropout)
        # self.emb_dropout = nn.Dropout(p)
        
        # Fastai embedding dropemb dropin
        self.emb_dropout = EmbeddingDropout(self.emb)
        self.dropouti = LockedDropout(drop_in)
        
    def forward(self, input):
        # Get embedding
        # x = self.emb_bn(self.emb_dropout(self.emb(input))) # Normal way to do
        x = self.emb_dropout(words=input, dropout=self.drop_emb if self.training else 0)
        x = self.dropouti(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class FC_Classifier(nn.Module):
    def __init__(self, n_input, n_output, drop_in = 0, hidden_szs = [2000, 1000], drops = [0.2, 0.2]):
        super().__init__()
        hidden_szs.insert(0, n_input)
        self.fin = nn.Sequential(nn.BatchNorm1d(n_input), nn.Dropout(drop_in))
        self.fcs = nn.ModuleList([nn.Sequential(nn.Linear(hidden_szs[i_layer], hidden_szs[i_layer+1]),
                                                nn.BatchNorm1d(hidden_szs[i_layer+1]),
                                                nn.Dropout(drops[i_layer]),
                                                nn.ReLU(True))
                                  for i_layer in range(len(hidden_szs)-1)])
        self.fout = nn.Linear(hidden_szs[-1], n_output)

    def forward(self, input):
        x = self.fin(input)
        for fc in self.fcs:
            x = fc(x)
        x = self.fout(x)
        return x

class CNNClassifier(nn.Module):
    def __init__(self):
        pass
    def forward(self, input):
        pass

# TODO:Upgrade to fastai RNN Encoder
class BiRNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers, bidirectional, drop_h=0.05, drop_weight=0):
        super().__init__()
        self.lstm = nn.LSTM(n_input, n_hidden, n_layers, 
                            batch_first=True, bidirectional=bidirectional, dropout=drop_h) # Dropout is crucial
        if drop_weight > 0:
            self.lstm = WeightDrop(self.lstm, dropout=drop_weight, weights=['weight_hh_l0'])
        
    def forward(self, input, initial_states):
        return self.lstm(input, initial_states)
    
# In comparision with other approaches
# https://arxiv.org/pdf/1509.01626.pdf (Character-level Convolution)
class MaxRNN(nn.Module):
    def  __init__(self, w2v, seqlen, n_labels, 
                  classifier_szs=[2000, 1000], classifier_drops=[0.2, 0.2],
                  hidden_size=300, n_layers=1, bidirectional=True,
                  drop_emb=0.02, drop_in=0.05, drop_rnnw=0.1, drop_rnnh=0.05):
        super().__init__()
        self.wordemb = WordEmbedding(w2v, seqlen, drop_emb=drop_emb, drop_in=drop_in)                                       
        self.hidden_size, self.n_labels, self.n_layers = hidden_size, n_labels, n_layers
        self.lstm = BiRNN(w2v.n_embfactors, hidden_size, n_layers, bidirectional, drop_h=drop_rnnh, drop_weight=drop_rnnw) 
        self.flat = Flatten()
        self.n_direction= 2 if bidirectional else 1
        self.classifier = FC_Classifier(n_input=hidden_size*self.n_direction, n_output=n_labels, 
                                        drop_in = 0, hidden_szs = classifier_szs, drops = classifier_drops)
        
    def forward(self, input):
        h0 = V(torch.zeros(self.n_layers*self.n_direction, input.shape[0], self.hidden_size))
        c0 = V(torch.zeros(self.n_layers*self.n_direction, input.shape[0], self.hidden_size))
        # Get embedding
        x = self.wordemb(input)
        # Forward propagate RNN
        x, _ = self.lstm(x, (h0, c0))
        last_layer = x[-1]
        x_max, _ = torch.max(x, dim=1)
        x = self.flat(x_max.contiguous())
        out = self.classifier(x)
        return F.softmax(out)

# Similar to FastAI PoolingLinearClassifier 
class PoolingRNN(nn.Module):
    def  __init__(self, w2v, seqlen, n_labels, 
                  classifier_szs=[1000], classifier_drops=[0.2],
                  hidden_size=300, n_layers=1, bidirectional=True,
                  drop_emb=0.02, drop_in=0.05, drop_rnnw=0.1, drop_rnnh=0.05):
        super().__init__()
        self.wordemb = WordEmbedding(w2v, seqlen, drop_emb=drop_emb, drop_in=drop_in)                                       
        self.hidden_size, self.n_labels, self.n_layers = hidden_size, n_labels, n_layers
        self.lstm = BiRNN(w2v.n_embfactors, hidden_size, n_layers, bidirectional, drop_h=drop_rnnh, drop_weight=drop_rnnw) 
        self.flat = Flatten()
        self.n_direction= 2 if bidirectional else 1
        self.classifier = FC_Classifier(n_input=hidden_size*self.n_direction*3, n_output=n_labels, 
                                        drop_in = 0, hidden_szs = classifier_szs, drops = classifier_drops)
    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1,2,0), (1,)).view(bs,-1)
    
    def forward(self, input):
        bs, seqlen = input.shape
        h0 = V(torch.zeros(self.n_layers*self.n_direction, bs, self.hidden_size))
        c0 = V(torch.zeros(self.n_layers*self.n_direction, bs, self.hidden_size))
        # Get embedding
        x = self.wordemb(input)
        
        # Forward propagate RNN
        outputs, _ = self.lstm(x, (h0, c0))
        # Use batch_first for RNN:  
        # output size: batch, seq, hidden_size * num_directions
        # hidden size: num_layers * num_directions, batch, hidden_size
        #
        last_layer = outputs[-1]
        #avgpool = self.pool(last_layer, bs, False)
        #mxpool = self.pool(last_layer, bs, True)
        #x = torch.cat([last_layer[-1], mxpool, avgpool], 1)
        avgpool = torch.mean(outputs, dim=1)
        mxpool, _ = torch.max(outputs, dim=1)
        lst = outputs[:, -1, :]
        x = torch.cat([lst, mxpool, avgpool], dim=1)
        # x = self.flat(x.contiguous())
        out = self.classifier(x)
        return F.softmax(out)

class MaxEmbedding(nn.Module): #80.51% p1=0.4; p2=0.3; p3=0.3; lr=1e-4 batch = 128
    def __init__(self, w2v, seqlen, n_labels, drop_emb=0.02, drop_in=0.05, drop_classifier=[0.2, 0.2, 0.2]):
        super().__init__()
        self.wordemb = WordEmbedding(w2v, seqlen, drop_emb=drop_emb, drop_in=drop_in)
        
        self.classifier = FC_Classifier(w2v.n_embfactors, n_labels, 
                                        drop_classifier[0], drop_classifier[1], drop_classifier[2])
        if init:
            init(self)
    def forward(self, input):
        x = self.wordemb(input)
        x = x.permute(0, 2, 1)
        #x = torch.mean(x, dim=2)
        x, _ = torch.max(x, dim=2)
        x = self.classifier(x)
        return F.softmax(x)
    
# LSTM with Attention network: 
#     http://colinraffel.com/publications/iclr2016feed.pdf
#     https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-RNN/
#     http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html (Good Pytorch example)
# 81.04% iter ~ 4250
# 81.43% iter ~ 6500
class AttentionBiRNN(nn.Module):
    def __init__(self, w2v, seqlen, n_labels, hidden_size, n_layers, p0=0.2, p1=0.2, p2=0.2, p3=0.2):
        super().__init__()
        self.wordemb = WordEmbedding(w2v, seqlen, p=p0)
        
        self.hidden_size, self.n_labels, self.n_layers = hidden_size, n_labels, n_layers
        self.lstm = nn.LSTM(w2v.n_embfactors, hidden_size, n_layers, batch_first=True, bidirectional=True, dropout=p1)
        self.flat = Flatten()
        
        self.classifier = FC_Classifier(hidden_size*2, n_labels, p1, p2, p3)
        self.attentioner = nn.Sequential(
            nn.BatchNorm1d(hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, 1)
            )
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, input):
        #Set initial state
        h0 = V(torch.zeros(self.n_layers*2, input.shape[0], self.hidden_size))
        c0 = V(torch.zeros(self.n_layers*2, input.shape[0], self.hidden_size))
        # Get embedding
        x = self.wordemb(input)
        # Forward propagate RNN
        x, _ = self.lstm(x, (h0, c0))
        x_flat_emb = x.contiguous().view(x.shape[0]*x.shape[1], -1)
        x_weight = self.attentioner(x_flat_emb)
        x_weight = x_weight.view(x.shape[0], x.shape[1], -1)
        x_weight = self.softmax(x_weight)
        x_attened = x_weight * x ### Broadcast at last dimension
        
        #x = torch.mean(x_attened, dim=1) # To Max?
        x, _ = torch.max(x_attened, dim=1) # To Max?
        #x    = x[:, -1, :] #Get last state
        x = self.flat(x.contiguous())
        out = self.classifier(x)
        return F.softmax(out)

# http://konukoii.com/blog/2018/02/19/twitter-sentiment-analysis-using-combined-lstm-cnn-models/
class CNNRNN(nn.Module):
    def __init__(self, w2v, seqlen, n_labels, 
                 hidden_size=300, n_layers=1, bidirectional=True, 
                 drop_emb=0.02, drop_in=0.05, drop_rnnw=0.1, drop_rnnh=0.05,
                 kernel_szs=[3,4,5], n_kernels=10,
                 classifier_szs=[1000], classifier_drops=[0.5]):
        super().__init__()
        self.wordemb = WordEmbedding(w2v, seqlen, drop_emb=drop_emb, drop_in=drop_in)
        self.n_direction= 2 if bidirectional else 1
        self.hidden_size, self.n_labels, self.n_layers = hidden_size, n_labels, n_layers
        self.lstm = BiRNN(w2v.n_embfactors, hidden_size, n_layers, bidirectional, drop_h=drop_rnnh, drop_weight=drop_rnnw) 
        self.flat = Flatten()
        
        self.convs = nn.ModuleList([
                        nn.Sequential(
                            nn.Conv1d(in_channels=seqlen, #Use Conv2d if LSTM n_layers and bidirectional? 
                                       out_channels=n_kernels, 
                                       kernel_size=kernel_sz),
                            nn.BatchNorm1d(n_kernels),
                            nn.Dropout(0.2),
                            nn.MaxPool1d(kernel_size = 3)
                        ) for kernel_sz in kernel_szs])
        self.classifier = FC_Classifier(n_input=self._get_conv_output(shape=(seqlen, hidden_size*self.n_direction)),
                                        n_output=n_labels, 
                                        drop_in = 0, hidden_szs = classifier_szs, drops = classifier_drops)        
    def forward(self, input):
        h0 = V(torch.zeros(self.n_layers*self.n_direction, input.shape[0], self.hidden_size))
        c0 = V(torch.zeros(self.n_layers*self.n_direction, input.shape[0], self.hidden_size))
        x = self.wordemb(input)
        x, _ = self.lstm(x, (h0, c0))
        # x = x.contiguous().view(x.shape[0]*x.shape[1], -1)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = self.classifier(self.flat(x))
        return F.softmax(x)
    
    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input = V(torch.rand(bs, *shape)).cpu()
        output_feat = self._forward_convs(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_convs(self, x):
        out = []
        for conv in self.convs:
            tmp = conv(x)
            out.append(tmp)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = self.flat(x)
        return x

def init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight)
            m.bias.data.zero_()