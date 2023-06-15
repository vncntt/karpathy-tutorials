import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


d = os.getcwd()
os.chdir("..")
os.getcwd()
words = open(os.getcwd()+'\\names.txt','r').read().splitlines()


#vocabulary and mappings to integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


block_size = 4
X,Y = [],[]
for w in words:
    context = [0]*block_size
    for ch in w+'.':
        idx = stoi[ch]
        Y.append(idx)
        X.append(context)
        #print(''.join(itos[i] for i in context),'---->',itos[idx])
        context = context[1:]+[idx]
X = torch.tensor(X)
Y = torch.tensor(Y)

emb_dim = 3
C = torch.randn((27,emb_dim))
W1 = torch.randn((block_size*emb_dim,200))
b1 = torch.randn(200)
W2 = torch.randn((200,27))
b2 = torch.randn(27)
parameters = [C,W1,b1,W2,b2]

for p in parameters:
    p.requires_grad = True



lri = []
lossi = []
for i in range(500000):
    #creating minibatch
    ix = torch.randint(0,X.shape[0],(32,))

    #forward pass
    emb = C[X[ix]]
    h = torch.tanh(emb.view(-1,emb_dim*block_size)@W1+b1)
    logits = h@W2 + b2
    loss = F.cross_entropy(logits,Y[ix])
    #print(loss.item())
    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    #update
    #lr = lrs[i]
    lr = 0.1 if i<10000 else 0.01 
    for p in parameters:
        p.data += -lr*p.grad

    #lri.append(lre[i])
    #lossi.append(loss.item())  
print(loss.item())



emb = C[X]
h = torch.tanh(emb.view(-1,emb_dim*block_size)@W1+b1)
logits = h@W2 + b2
loss = F.cross_entropy(logits,Y)
print('real loss: ',loss)



#generating actual names

output = []

for _ in range(10):
    out = []
    context = block_size*[0]
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1,-1)@W1+b1)
        logits = h@W2 + b2
        probs = F.softmax(logits,dim=1)
        idx = torch.multinomial(probs,num_samples=1).item()
        context = context[1:]+[idx]
        out.append(idx)
        if idx==0:
            break
    print(''.join(itos[i] for i in out))