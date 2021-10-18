import torch
'''
#a=torch.zeros((3,4))
b=torch.ones((3,1))
print(b)
b=b.repeat((1,4))
print(b)
#print(a,b,a+b)
'''
a=torch.randn((2,3))
b=torch.randn((3,4,5))
print()

a=torch.ones((3,6,1))
b=a.repeat((1,1,4))
print(b.shape)
print(b)