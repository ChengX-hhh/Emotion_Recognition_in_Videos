
from torch import nn
from torch.nn import functional as F
import torch
from config import DefaultConfig
import torchvision.models as models

class FineTuneModel(nn.Module):
    
    def __init__(self, num_classes=2):
        super(FineTuneModel, self).__init__()
        # self.model_name = 'ALDI_TI'
        self.opt = DefaultConfig()


        self.resnet34 = models.resnet34(pretrained=True)
        # self.resnet34 = models.resnet34()
        # print (self.resnet34)
        # return
        self.resnet34.fc = nn.Linear(512, 300) 
        self.resnet34.avgpool = nn.AvgPool2d(4, stride=1)
        # print (self.resnet34)

        self.fc = nn.Linear(300, 30) 
        self.fc2 = nn.Linear(30, 7) 
        
    def layer1(self,X):

        image_rep = self.resnet34(X) #(batch,300)
        # print (image_rep.shape)
        output = F.tanh(self.fc(image_rep))
        output = F.tanh(self.fc2(output))
        result = F.softmax(output) #(batch,7)
        return result



    def forward(self, input):
 

        output = self.layer1(input)

        return F.softmax(output)
