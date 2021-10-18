import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import math
def reset_parameters(weight):
    stdv=1./math.sqrt(weight.size(1))
    weight.data.uniform(-stdv,stdv)
    return weight

class MatMulLayer(nn.Module):
    def __init__(self,dim0,dim1):
        super(MatMulLayer, self).__init__()
        self.dim0=dim0
        self.dim1=dim1
        self.weight=nn.Parameter(torch.empty(dim0,dim1))
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform(self.weight,a=math.sqrt(5))
    def forward(self,input):
        return torch.matmul(self.weight,input)

class LSTMWithAttention(nn.Module):
    def __init__(self,conf):
        super(LSTMWithAttention, self).__init__()
        self.conf=conf
        self.embedding=nn.Embedding(conf.vocab_size,conf.embedding_dim).from_pretrained(conf.pretrained_vectors,freeze=True)#以后不更新参数
        #定义两个LSTM，第一个LSTM的输入cn作为第二个LSTM的输入c0
        self.lstm1=nn.LSTM(conf.embedding_dim,conf.hidden_dim,conf.lstm_layers,batch_first=True)
        self.lstm2=nn.LSTM(conf.embedding_dim,conf.hidden_dim,conf.lstm_layers,batch_first=True)
        #W_y,W_h,
        '''
        self.W_y=nn.Parameter(torch.Tensor(conf.hidden_dim,conf.hidden_dim))
        self.W_h=nn.Parameter(torch.Tensor(conf.hidden_dim,conf.hidden_dim))
        self.w=nn.Parameter(torch.Tensor(1,conf.hidden_dim))
        self.W_y=reset_parameters(self.W_y)
        self.W_h=reset_parameters(self.W_h)
        self.w=reset_parameters(self.w)
        '''
        self.W_y=MatMulLayer(conf.hidden_dim,conf.hidden_dim)
        self.W_h = MatMulLayer(conf.hidden_dim, conf.hidden_dim)
        self.w = MatMulLayer(1, conf.hidden_dim)
        self.fc1=nn.Linear(conf.hidden_dim,conf.hidden_dim)
        self.fc2=nn.Linear(conf.hidden_dim,conf.hidden_dim)
        self.fc3=nn.Linear(conf.hidden_dim,conf.num_classes)

    def attention(self,q,V):#这里的q是查询向量，V是输入向量
        #q=(batch_size,hidden_dim)
        #V=(batch_size,sentence_length,hidden_dim)
        V_t=V.permute(0,2,1)#(batch_size,hidden_dim,sentence_length)
        sentence_length=V_t.shape[2]
        a1=self.W_y(V.permute(0,2,1))
        #a1=torch.matmul(self.W_y,V.permute(0,2,1))#(batch_size,hidden_dim,sentence_length)
        #m=q.unsqueeze(2)
        a2=self.W_h(q.unsqueeze(2))
        #a2=torch.matmul(self.W_h,q.unsqueeze(2))#(batch_size,hidden_dim,1)
        a2=a2.repeat((1,1,sentence_length))#(batch_size,hidden_dim,sentence_length)
        weight=F.tanh(a1+a2)#(batch_size,hidden_dim,sentence_length)
        weight=self.w(weight)
        #weight=torch.matmul(self.w,weight)#(batch_size,1,sentence_length)
        scores=F.softmax(weight,dim=2)#(batch_size,1,sentence_length)
        r=torch.matmul(V.permute(0,2,1),scores.permute(0,2,1))
        #(batch_size,hidden_dim,sentence_length) * (batch_size,sentence_length,1)
        #r=(batch_size,hidden_dim,1)
        return r

    def forward(self,input):#输入为(batch_size,sentence_length)
        premise_input=input[0]
        hypothesis_input=input[1]
        x=self.embedding(premise_input)#(batch_size,sentence_length,embedding_dim)
        output_p,(hn_p,cn_p)=self.lstm1(x)
        #output_p=(batch_size,sentence_length,hidden_dim)
        #hn_p=(num_layer,batch_size,hidden_dim)
        #cn_p=(num_layer,batch_size,hidden_dim)
        h0=torch.zeros((cn_p.size())).cuda()#将其放到cuda上
        hypothesis_x=self.embedding(hypothesis_input)#同样也需要对hypothesis进行embedding
        output_hypo,(hn_hypo,cn_hypo)=self.lstm2(hypothesis_x,(h0,cn_p))
        #output_hypo=(batch_size,sentence_length,hidden_dim)
        #hn_hypo=(num_layer,batch_size,hidden_dim)
        #cn_hypo=(num_layer,batch_size,hidden_dim)
        #接下来使用output_p和hn_hypo来进行attention的计算
        hn_hypo=hn_hypo.squeeze(0)#(batch_size,hidden_dim)#这是假设维度为1的情况
        cn_hypo=cn_hypo.squeeze(0)#(batch_size,hidden_dim)
        attn=self.attention(hn_hypo,output_p)#(batch_size,hidden_dim,1)
        attn=attn.squeeze(2)#(batch_size,hidden_dim)
        representation=F.tanh(self.fc1(attn)+self.fc2(hn_hypo))#(batch_size,hidden_dim)
        logit=self.fc3(representation)#(batch_size,num_classes)
        return logit