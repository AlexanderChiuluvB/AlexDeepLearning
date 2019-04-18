#coding:utf8
import torch
import time

class BasicModule(torch.nn.Module):
    """
    模型封装
    """
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self))

    def load(self,path):
        self.load_state_dict(torch.load(path))

    def save(self,name=None):

        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix+'%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(),name)
        return name

    def get_optimizer(self,lr,weight_decay):
        return torch.optim.Adam(self.parameters(),lr=lr,weight_decay=weight_decay)

class Flat(torch.nn.Module):

    def __init__(self):
        super(Flat,self).__init__()

    def forward(self,x):
        """
        把输入 reshape (batch_size,dim_length)
        :param x:
        :return:
        """
        return x.view(x.size(0),-1)
