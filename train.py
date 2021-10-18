import os
os.environ['CUDA_VISIBLE_DEVICES']="0,2,3,4"
import data
from models.LSTMWithAttention import LSTMWithAttention
import torch
import torch.nn.functional as F
class Config(object):
    vocab_size=0#后续更新
    embedding_dim=300
    hidden_dim=200
    lstm_layers=1
    num_classes=0#后续更新
    pretrained_vectors=None#表示预训练的词向量

    cuda=True
    epochs=200
    batch_size=64
    shuffle=True
    learning_rate = 0.001
    learning_momentum = 0.9
    weight_decay = 0.0001
    data_path='data/snli_1.0'


def train(train_iter,model,conf):
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    model.train()
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    if conf.cuda:
        model = torch.nn.DataParallel(model).cuda()

    best_acc = 0
    for epoch in range(conf.epochs):
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        batch_cnt = 0
        for batch in train_iter:
            premise=batch.premise
            hypothesis=batch.hypothesis
            target=batch.label
            premise=premise.t()
            hypothesis=hypothesis.t()
            target = torch.sub(target, 1)  # 让class从0开始
            if conf.cuda:
                premise=premise.cuda()
                hypothesis=hypothesis.cuda()
                target=target.cuda()

            optimizer.zero_grad()
            input_data=[premise,hypothesis]
            logit=model(input_data)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            pred = (torch.max(logit, 1))[1].view(target.size())  # 表示预测出的标签
            correct = (pred.data == target.data).sum()  # 得到的是一个只有一个元素的tensor
            total_loss += loss.item()
            total_correct += correct.item()
            total_count += batch.batch_size
            assert premise.shape[0] == batch.batch_size
            batch_cnt += 1

        total_loss /= total_count
        acc = total_correct / total_count

        print('Training epoch [%d/%d] - training loss: %.6f  '
              'training acc: %.4f' % (
              epoch, conf.epochs, total_loss, acc))


if __name__=="__main__":
    conf=Config()
    train_iter,text_field,label_field=data.snli_prepare(conf.data_path,conf.batch_size,shuffle=True,kind="train")
    test_iter,_,_=data.snli_prepare(conf.data_path,conf.batch_size,shuffle=False,kind="test")

    conf.pretrained_vectors=text_field.vocab.vectors

    conf.vocab_size=len(text_field.vocab)
    conf.num_classes=len(label_field.vocab)-1
    model=LSTMWithAttention(conf)
    #for i in model.parameters():
    #    print(i)
    train(train_iter,model,conf)

