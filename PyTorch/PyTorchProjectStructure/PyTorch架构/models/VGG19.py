from .BasicModule import BasicModule
from torch import nn
import torch.utils.model_zoo as model_zoo


class VGG19(BasicModule):

    """

    VGG19 with batchNorm

    """
    def __init__(self,features,num_classes=20):
        super(VGG19,self).__init__()

        self.features = features
        self.classifier = nn.Sequential(

            #output layer
            nn.Linear(6144,1024),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(1024,20),


        )
        #初始化参数
        self._init_weight()

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



model_cfg = {

    'VGG19_BN': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

}

def make_layers(model_cfg):
    layers = []
    input_channel = 1

    for l in model_cfg:
        if l=='M':
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            conv2d = nn.Conv2d(input_channel,l,kernel_size=3,padding=1)
            layers += [conv2d,nn.BatchNorm2d(l),nn.ReLU(inplace=True)]
            input_channel = l

    return nn.Sequential(*layers)



def vgg19_bn(pretrained=False,**kwargs):

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG19(make_layers(model_cfg['VGG19_BN']),**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'))
    return model
