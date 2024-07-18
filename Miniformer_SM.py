import torch
import math
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from d2l import torch as d2l
import scipy.io as sio
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import random
import scipy.io
import h5py
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pdb
from sklearn.metrics import roc_curve, auc
from torchinfo import summary 
import time 
from tqdm import tqdm 

torch.manual_seed(123)  
np.random.seed(123)
random.seed(123)

def up_triu(X, k=0):
    mask = torch.ones(X.shape[1], X.shape[2])
    mask = torch.nonzero(torch.triu(mask, k)).t()
    r = []
    for x in X:
        uptri = torch.triu(x, k)
        r.append(uptri)
    return torch.stack(r, dim=0), mask

def flatten(X, mask):
    r = []
    for x in X:
        x = x[mask[0], mask[1]]
        r.append(x)
    return torch.stack(r, dim=0)

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)


class DotProductAttention(nn.Module):
    """Scaled dot product attention."""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        self.scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = nn.Softmax(dim=-1)(self.scores)
        return torch.matmul(self.attention_weights, values), self.scores 


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.w = nn.Parameter(torch.randn(1, query_size))

    def forward(self, queries, keys, values):
        self.w_diag = torch.diag(self.w.squeeze())

        q_s = torch.matmul(queries, self.w_diag)
        k_s = torch.matmul(keys, self.w_diag)
        v_s = torch.matmul(values, self.w_diag)

        queries = transpose_qkv(q_s, self.num_heads)
        keys = transpose_qkv(k_s, self.num_heads)
        values = transpose_qkv(v_s, self.num_heads)

        output, score = self.attention(queries, keys, values)  
        output_concat = transpose_output(output, self.num_heads)

        return output_concat, score, self.w  

def transpose_qkv(x, num_heads):
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    x = x.permute(0, 2, 1, 3)

    return x.reshape(-1, x.shape[2], x.shape[3])


def transpose_output(x, num_heads):
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    x = x.permute(0, 2, 1, 3)
    return x.reshape(x.shape[0], x.shape[1], -1)


class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, x):
        xx, scores, w = self.attention(x, x, x)
        y = self.addnorm1(x, xx)
        return self.addnorm2(y, self.ffn(y)), scores, w


class TransformerEncoder(d2l.Encoder):
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))

    def forward(self, x, *args):
        self.attention_weights = [None] * len(self.blks)
        self.scores = [None] * len(self.blks)
        W = []
        for i, blk in enumerate(self.blks):
            x, score, w = blk(x)
            W.append(w)
            self.attention_weights[i] = blk.attention.attention.attention_weights  
            self.scores[i] = blk.attention.attention.scores
        return x, self.scores, self.attention_weights, W 

class TransformerModel(nn.Module):
    def __init__(self, key_size, query_size, value_size, hidden_size, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, num_classes, dropout):
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(key_size, query_size, value_size, hidden_size,
                                          norm_shape, ffn_num_input, ffn_num_hiddens,
                                          num_heads, num_layers, dropout)  
        self.fc = nn.Linear(116, num_classes) 

        self.fc1 = nn.Linear(6786,256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropoutfc = nn.Dropout(p=0.1)

    def forward(self, x):
        x, scores, attention_weights, W = self.encoder(x)  
        x = scores[-1]  
        x, mask = up_triu(x,k=0) 
        x = flatten(x, mask)
        x = self.fc1(x)  
        x = self.relu(x)  
        x = self.dropoutfc(x)  
        x = self.fc2(x)  
        return x, W


class Fmri(object):
    def read_data(self):  
        dataFile = r'path：'
        data = scipy.io.loadmat(dataFile)
        AAL = data['AAL']
        y = np.squeeze(data['lab'])
        X = np.empty([184, 175, 116])
        for i in range(184):
            X[i] = AAL[0, i]
        X = torch.tensor(X).float()  
        X = X.permute(0, 2, 1)  
        return X, y

    def __init__(self):  
        super(Fmri, self).__init__()
        X, y = self.read_data()
        self.X = X
        self.y = torch.from_numpy(y)
        self.n_samples = X.shape[0]


    def __getitem__(self, index): 
        return self.X[index], self.y[index]

    def __len__(self):  
        return self.n_samples
    

from sklearn.model_selection import KFold

full_dataset = Fmri()  
num_folds = 10
kfold = KFold(n_splits=num_folds, shuffle=True)
avg_accuracy = 0.0  


key_size = 175
query_size = 175
value_size = 175
hidden_size = 175 
norm_shape = 175
ffn_num_input = 175
fn_num_hiddens = 16  
num_heads = 1
num_layers = 2
num_classes = 2
dropout = 0.1
epochs = 50
lamda=0.5
miu=0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def adjacent_difference2(d_v):
    d_v = d_v.squeeze()
    n = len(d_v)
    total_diff = 0
    
    for i in range(0, n):
        b = 0  
        for j in range(0, n):
            if j != i:
                b += d_v[j]
        diff = d_v[i] - (b / (n - 1))
        total_diff += diff    
    return total_diff  
    
def adjacent_difference(d_v):
    d_v = d_v.squeeze()
    n = len(d_v)  
    total_diff = 0
    for i in range(1, n):
        diff = (d_v[i] - d_v[i-1])**2
        total_diff += diff    
    return total_diff   


model_init = TransformerModel(key_size, query_size, value_size, hidden_size, norm_shape, ffn_num_input, fn_num_hiddens,
                          num_heads, num_layers, num_classes, dropout).to(device)
para = model_init.state_dict()
SEN,SPE,BAC,PPV,NPV,PRE,REC,F1score = [],[],[],[],[],[],[],[]
AUC_scores = []
Label = []
Prob = []
Time = [] 

for fold, (train_indices, test_indices) in enumerate(kfold.split(full_dataset)):
    train_set = torch.utils.data.Subset(full_dataset, train_indices)
    test_set = torch.utils.data.Subset(full_dataset, test_indices)

    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=8, shuffle=True)
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=8, shuffle=True)

    model = TransformerModel(key_size, query_size, value_size, hidden_size, norm_shape, ffn_num_input, fn_num_hiddens,
                             num_heads, num_layers, num_classes, dropout).to(device)
    model.load_state_dict(para)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)      
    TR_A, TR_L, TE_A, TE_L = [], [], [], []      
    for epoch in range(epochs):
        train_acc = 0.0  
        train_loss = 0.0
        test_acc = 0.0
        test_loss = 0.0
        true_labels = []
        pred_labels = []         
        model.train() 
        D_v0, D_v1 = 0.0, 0.0
        L2_0, L2_1 = 0.0, 0.0
        
        start=time.time()
        for i, data in enumerate(tqdm(train_loader,ncols=80,desc=f'Training | Fold:{fold + 1},Epoch:{epoch+1}')):
            x, y = data  
            x = x.to(torch.float32)  
            y = y.to(torch.float32)
            x = x.to(device)  
            y = y.to(device)
            optimizer.zero_grad()  
            output = model(x)  
            _p0 = output[1][0]
            _p1 = output[1][1]
            
            
            l2_0 = torch.norm(_p0, p=2)
            l2_1 = torch.norm(_p1, p=2)
            L2_0 = l2_0
            L2_1 = l2_1
            
            d_v0 = adjacent_difference(_p0)
            d_v1 = adjacent_difference(_p1)
            
            D_v0 = d_v0
            D_v1 = d_v1

            
            batch_loss = criterion(output[0], y.long()) 
            _, train_pred = torch.max(output[0], 1)  
          
            batch_loss = batch_loss+ miu*(l2_0 + l2_1) +lamda*(d_v0 + d_v1) 
            
            
            batch_loss.backward() 
            optimizer.step() 
            
            train_acc += (train_pred.cpu() == y.cpu()).sum().item()  
            train_loss += batch_loss.item()  
            
        end=time.time()
        training_time=end-start
        Time.append(training_time)
        
        train_acc /= len(train_loader.dataset) 

        model.eval()  
        if epoch == epochs - 1: PROB = []
        with torch.no_grad():  
            D_v0 = D_v0.detach()
            D_v1 = D_v1.detach()
            for i, data in enumerate(tqdm(test_loader,ncols=80,desc=f'Testing  | Fold:{fold + 1},Epoch:{epoch+1}')):
                x, y = data
                x = x.to(torch.float32)
                y = y.to(torch.float32)
                x = x.to(device)
                true_labels.extend(y.numpy())
                y = y.to(device)

                output = model(x)

                if epoch == epochs - 1:
                    prob = output[0].softmax(1).tolist()
                    PROB.extend(prob)
                
                batch_loss = criterion(output[0], y.long())
                batch_loss = batch_loss+ miu*(l2_0 + l2_1) +lamda*(D_v0 + D_v1) 
        
                _, test_pred = torch.max(output[0], 1)  
                pred_labels.extend(test_pred.cpu().numpy())
                
                test_acc += (test_pred.cpu() == y.cpu()).sum().item()  
                test_loss += batch_loss.item()

            test_acc /= len(test_loader.dataset)  

            if epoch == epochs - 1:
                Label.extend(true_labels)
                Prob.extend(PROB)
            
         
            print(
                f"Fold {fold + 1} Epoch {epoch + 1} Train Accuracy: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
     
        TR_A.append(train_acc)
        TR_L.append(train_loss / 5000)
        TE_A.append(test_acc)
        TE_L.append(test_loss / 5000)


    x_axis = np.arange(epochs)
    plt.plot(x_axis, TR_A, '-b', label='train_acc')
    plt.plot(x_axis, TR_L, '-r', label='train_loss')
    plt.plot(x_axis, TE_A, '-g', label='test_acc')
    plt.plot(x_axis, TE_L, '-c', label='test_loss')
    plt.title('fold '+str(fold))
    plt.legend()
    plt.show()
  

    
    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    sen = tp / (tp + fn)  
    spe = tn / (tn + fp)  
    bac = (sen + spe) / 2  
    ppv = tp / (tp + fp)  
    npv = tn / (tn + fn)  
    pre = tp / (tp + fp)  
    rec = tp / (tp + fn)  
    f1score = 2 * (pre * rec) / (pre + rec)  
   
    print("（sen）:", sen, end=" ")
    print("（spe）:", spe, end=" ")
    print("（bac）:", bac, end=" ")
    print("（ppv）:", ppv)
    print("（npv）:", npv, end=" ")
    print("（pre）:", pre, end=" ")
    print("（rec）:", rec, end=" ")
    print("（f1score）:", f1score)
    
    SEN.append(sen)
    SPE.append(spe)
    BAC.append(bac)
    PPV.append(ppv)
    NPV.append(npv)
    PRE.append(pre)
    REC.append(rec)
    F1score.append(f1score)
    
    avg_accuracy += test_acc  
avg_accuracy /= num_folds  
print(f"Average Accuracy: {avg_accuracy:.4f}")  


SEN = sum(SEN) / num_folds
SPE = sum(SPE) / num_folds
BAC = np.mean(BAC)
PPV = np.mean(PPV)
NPV = np.mean(NPV)
PRE = np.mean(PRE)
REC = np.mean(REC)
F1score = np.mean(F1score)

print("敏感性（SEN）:", SEN)
print("特异性（SPE）:", SPE)
print("平衡准确率（BAC）:", BAC)
print("阳性预测值（PPV）:", PPV)
print("阴性预测值（NPV）:", NPV)
print("精确率（PRE）:", PRE)
print("召回率（REC）:", REC)
print("F1得分（F1score）:", F1score)

