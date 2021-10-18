import torchtext
from torchtext.data import Field,Dataset,Example
from torchtext.vocab import Vectors
import pickle
import os
import numpy as np
import json
def text_tokenize(x):
    return [w for w in x.split(" ") if len(w)>0]

def label_tokenize(x):#将一个标签转化为了一个shape为1的数组
    return [x.replace("__label__","")]

class SNLI_Dataset(Dataset):
    def __init__(self,path,text_field,label_field):
        fields=[('premise',text_field),('hypothesis',text_field),('label',label_field)]
        examples=[]
        #content=np.loadtxt(path,dtype=str,delimiter='\t',skiprows=1,usecols=[0,5,6])
        with open(path,'r') as f:
            for item in f:
            #for item in content:#对于每一个样本，都构建一个Example
                item=json.loads(item)
                premise=item['sentence1']
                hypothesis=item['sentence2']
                label=item['gold_label']
                e=Example.fromlist([premise,hypothesis,label],fields)
                examples.append(e)
        super(SNLI_Dataset, self).__init__(examples,fields)

    @staticmethod
    def sort_key(ex):
        return len(ex.premise)

def save_vocab(vocab,filename):
    with open(filename,'wb') as f:
        pickle.dump(vocab,f)
def load_vocab(filename):
    with open(filename, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def snli_prepare(path,batch_size,shuffle=False,kind='train'):
    text_field=torchtext.data.Field(sequential=True, tokenize=text_tokenize, lower=True)
    label_field = torchtext.data.Field(sequential=False, tokenize=label_tokenize, lower=True)
    assert kind in {'train','test','dev'}
    filename='snli_1.0_'+kind+'.jsonl'
    dataset=SNLI_Dataset(os.path.join(path,filename),text_field,label_field)
    if kind=="train":
        #TODO 这里的地址需要改一下，并且看看有没有word2vec的词向量
        glove_vectors = Vectors(name='/home/songsihan/NLP/pretrainedVectors/glove/glove.6B.300d.txt',
                                cache='/home/songsihan/NLP/pretrainedVectors/glove/')
        text_field.build_vocab(dataset,vectors=glove_vectors)
        label_field.build_vocab(dataset)
        if not os.path.exists('vocab/'):#保留构建的词汇表
            os.makedirs('vocab/')
        save_vocab(text_field.vocab, 'vocab/text.vocab')  # 这里保存词汇表是因为词汇表在test过程中会用到
        save_vocab(label_field.vocab, 'vocab/label.vocab')
        train_iterator = torchtext.data.BucketIterator(dataset, batch_size,
                                                       sort_key=lambda x: len(x.premise),
                                                       shuffle=shuffle)  # 这里的这个text应该是和上面Dataset中Field的名字是相对应的。
        return train_iterator, text_field, label_field
    else:
        text_field.vocab = load_vocab('vocab/text.vocab')
        label_field.vocab = load_vocab('vocab/label.vocab')
        test_iterator = torchtext.data.BucketIterator(dataset, batch_size,
                                                      sort_key=lambda x: len(x.text),
                                                      shuffle=False)
        return test_iterator, text_field, label_field
