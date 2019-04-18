"""
配置训练的变量

"""


class DefaultConfig(object):

    env = 'default' #visdom environment
    model = 'AlexNet'

    train_data_root = './train'
    test_data_root = './test1'
    load_model_path = 'checkpoints/model.pth'

    batch_size = 128
    use_gpu = True
    num_workers = 4 # workers for loading data
    print_freq = 20 #print info every N batch

    debug_file = '/tmp/debug'
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.1
    lr_decay = 0.95
    weight_decay = 1e-4 #损失函数


def parse(self,kwargs):
    """

    通过传入字典来覆盖参数，不需要每次都修改config.py
    :param kwargs:
    :return:

    """

    for k,v in kwargs.items():
        if not hasattr(self,k):
            return
        setattr(self,k,v)

    "print info"
    for k,v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k,getattr(self,k))



DefaultConfig.parse = parse
opt = DefaultConfig()

