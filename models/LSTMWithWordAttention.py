import torch
import torch.nn as nn
import torch.nn.functional as F
from LSTMWithAttention import MatMulLayer
class LSTMWithWordAttention(nn.Module):
    def __init__(self, conf):
        super(LSTMWithWordAttention, self).__init__()
        self.conf = conf
        self.embedding = nn.Embedding(conf.vocab_size, conf.embedding_dim).from_pretrained(conf.pretrained_vectors,
                                                                                           freeze=True)  # 以后不更新参数
        # 定义两个LSTM，第一个LSTM的输入cn作为第二个LSTM的输入c0
        self.lstm1 = nn.LSTM(conf.embedding_dim, conf.hidden_dim, conf.lstm_layers, batch_first=True)
        self.lstm2 = nn.LSTM(conf.embedding_dim, conf.hidden_dim, conf.lstm_layers, batch_first=True)

        self.W_y=MatMulLayer(conf.hidden_dim,conf.hidden_dim)
        self.W_h=MatMulLayer(conf.hidden_dim,conf.hidden_dim)
        self.W_r=MatMulLayer(conf.hidden_dim,conf.hidden_dim)
        self.w=MatMulLayer(1,conf.hidden_dim)
        self.W_t=MatMulLayer(conf.hidden_dim,conf.hidden_dim)

        self.fc1=nn.Linear(conf.hidden_dim,conf.hidden_dim)
        self.fc2=nn.Linear(conf.hidden_dim,conf.hidden_dim)
        self.fc3=nn.Linear(conf.hidden_dim,conf.num_classes)

    def forward(self,input):
        premise_input = input[0]
        hypothesis_input = input[1]
        x = self.embedding(premise_input)  # (batch_size,sentence_length,embedding_dim)
        output_p, (hn_p, cn_p) = self.lstm1(x)
        # output_p=(batch_size,sentence_length,hidden_dim)
        # hn_p=(num_layer,batch_size,hidden_dim)
        # cn_p=(num_layer,batch_size,hidden_dim)
        h0 = torch.zeros((cn_p.size())).cuda()  # 将其放到cuda上
        hypothesis_x = self.embedding(hypothesis_input)  # 同样也需要对hypothesis进行embedding
        output_hypo, (hn_hypo, cn_hypo) = self.lstm2(hypothesis_x, (h0, cn_p))
        #接下来根据output_p和output_hypo来求每个h_i的注意力r_i
        #output_hypo=(batch_size,sentnece_length,hidden_dim)
        attn=self.attention(output_hypo,output_p)#(batch_size,hidden_dim,1)
        attn=attn.squeeze(2)
        representation = F.tanh(self.fc1(attn) + self.fc2(hn_hypo))  # (batch_size,hidden_dim)
        logit = self.fc3(representation)  # (batch_size,num_classes)
        return logit

    def attention(self,Q,V):
        #Q=(batch_size,sentence_length2,hidden_dim) 即h
        #V=(batch_size,sentence_length1,hidden_dim) 即Y
        a1=self.W_y(V.permute(0,2,1))#(batch_size,hidden_dim,sentence_length1)
        hypothesis_len=Q.shape[1]
        premise_len=V.shape[1]
        r=torch.zeros((Q.shape[0],Q.shape[2],1))#(batch_size,hidden_dim,1)
        for i in range(hypothesis_len):
            a2=self.W_h(Q[:,i,:].permute(0,2,1))+self.W_r(r)#(batch_size,hidden_dim,1)
            a2=a2.repeat((1,1,premise_len))
            weight=F.tanh(a1+a2)#(batch_size,hidden_dim,sentence_length1)
            weight=self.w(weight)#(batch_size,1,sentence_length1)
            scores=F.softmax(weight,dim=2)
            r=torch.matmul(V.permute(0,2,1),scores.permute(0,2,1))+F.tanh(self.W_t(r))#计算出下一步的r
        #得到了最终计算出的r
        return r

