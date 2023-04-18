class DefaultConfig(object):
    """docstring for DefaultConfig"""
    def __init__(self):
        super(DefaultConfig, self).__init__()
        
        self.train_data_root = 'data/fer2013/train'
        self.val_data_root = 'data/fer2013/val'
        self.test_data_root = 'data/fer2013/test'

        self.batch_size = 512
        self.num_workers = 4


        self.max_epoch = 100
        self.lr = 1e-4
        self.lr_decay = 0.95
        self.weight_decay = 1e-4





