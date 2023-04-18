import models
import torch
from data import FER2013
from config import DefaultConfig
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score,classification_report,confusion_matrix
import time
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms as T
np.set_printoptions(threshold=np.inf)

#os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():

    opt = DefaultConfig()

    model = getattr(models, 'FineTuneModel')()
    print (model)
  
    #model = torch.nn.DataParallel(model)
    model = model.cuda()
    #model = model.to(device)
    

    train_data = FER2013(opt.train_data_root,opt)
    val_data = FER2013(opt.val_data_root, opt)
    test_data = FER2013(opt.test_data_root, opt)

    print (train_data.__getitem__(0))
    

    train_dataloader = DataLoader(
        train_data,
        opt.batch_size,
        shuffle = True,
        num_workers = 4
        )
    
    print ('train_dataloader')
    val_dataloader = DataLoader(
        val_data,
        opt.batch_size,
        shuffle = False,
        num_workers = 4
        )
    
    print ('val_dataloader')
    test_dataloader = DataLoader(
        test_data,
        opt.batch_size,
        shuffle = False,
        num_workers = 4
        )

    print ('test_dataloader')
    

    # class_dis = [15,9]
    # sum_num = sum(class_dis)
    # weight = torch.Tensor([sum_num/x/opt.num_classes for x in class_dis]).cuda()

    # criterion = torch.nn.CrossEntropyLoss(weight=weight)


    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(),
        lr = lr,
        weight_decay = opt.weight_decay)
    writer = SummaryWriter()
    cnt=0

    sum_acc, sum_loss = 0.,0.



    for epoch in range(opt.max_epoch):
        for i, (input, target) in tqdm(enumerate(train_dataloader),total=len(train_data)/opt.batch_size):
            
            # print (input, target)
            # print ("shape ", input.shape, target.shape)
          
 
            input = input.cuda()
            #input = input.to(device)
            target = target.cuda()
            #target = target.to(device)

            # print (type(input1),input1.shape)
            optimizer.zero_grad()
            # print (input1.type(),input2.type(),sentence_len.type(),sample_len.type(),target.type())
            score = model(input)


            loss = criterion(score,target)
            loss.backward()

            optimizer.step()


            y_ = torch.argmax(score,dim = 1).cpu().numpy()
            correct = sum(y_==target.cpu().numpy())/opt.batch_size
            # print ('pred ',y_)
            # print ('target ',target.cpu().numpy())
            sum_acc += correct
            sum_loss += loss.data
            # sum_acc_ += correct
            # sum_loss_ += loss.data

            print("epoch {} iter {} sum correct {} loss {}".format(epoch, i,sum_acc/(i+1),sum_loss/(i+1)))
            writer.add_scalar("/train/acc",sum_acc/(i+1),cnt)
            writer.add_scalar("/train/loss",sum_loss/(i+1),cnt)

            cnt +=1

            
        sum_acc, sum_loss = 0., 0.

        with torch.no_grad():
            print ("=================================val======================================")
            val_test(model, val_dataloader, len(val_data), opt.batch_size, writer, epoch, "/val/")
            print ("=================================test======================================")
            val_test(model, test_dataloader, len(test_data), opt.batch_size, writer, epoch, "/test/")

        torch.save(model.state_dict(), 'checkpoints/ResNet'+str(epoch)+'.bin')
    writer.close()
  

def val_test(model,dataloader, lenn, batch_size, writer,epoch, dataset):
    model.eval()
    sum_acc = 0.
    sum_y_, sum_target, logits = np.asarray([]), np.asarray([]), np.asarray([])
    for i, (input, target) in tqdm(enumerate(dataloader), total = lenn/batch_size):

        input = input.cuda()
        #input = input.to(device)
        target = target.cuda()
        #target = target.to(device)

        score = model(input)
        print (input.shape)
        y_ = torch.argmax(score,dim = 1).cpu().numpy()
        correct = sum(y_==target.cpu().numpy())

        sum_acc += correct

        sum_y_ =  np.concatenate((sum_y_,y_),axis=0)
        sum_target = np.concatenate((sum_target,target.cpu().numpy()),axis=0)

    print (sum_y_)
    print (sum_target)

    f1 = metrics(sum_target,sum_y_)

    writer.add_scalar(dataset + 'acc', sum_acc/lenn, epoch)
    writer.add_scalar(dataset + 'f1', f1, epoch)


    print("correct {} ".format(sum_acc/lenn))
    print("f1 {} ".format(f1))

    
    model.train()


def metrics(y_true,y_pred):


    # print(precision_score(y_true, y_pred, average='macro'))
    # print(recall_score(y_true, y_pred, average='macro'))
    f1 = f1_score(y_true, y_pred, average='macro')
    print('macro ',f1_score(y_true, y_pred, average='macro'))
    print('micro ', f1_score(y_true, y_pred, average='micro'))
    # print('binary 1 ',f1)

    print (confusion_matrix(y_true, y_pred))
    target_names = ['0', '1','2', '3','4', '5','6']
    print(classification_report(y_true, y_pred, target_names=target_names))
    return f1


def use_model(*face_ims):
    train_transform = T.Compose([
            T.Resize((128, 128)),  # 缩放
            # transforms.RandomCrop(32, padding=4),  # 随机裁剪
            T.ToTensor(),  # 图片转张量，同时归一化0-255 ---》 0-1
            T.Normalize(0, 1),  # 标准化均值为0标准差为1
        ])


    model = getattr(models, 'FineTuneModel')()
    state_dict = torch.load('checkpoints/ResNet78.bin')
    #state_dict = torch.load('checkpoints/ResNet78.bin',map_location='cuda:0')
    # print (state_dict1.keys())
    for k in list(state_dict.keys()):
        state_dict[k[7:]] =  state_dict.pop(k)
    
    # 导入已训练好的模型
    model.load_state_dict(state_dict)
    model.eval()
    #model = torch.nn.DataParallel(model)
    model = model.cuda()
    #model = model.to(device)
    
    
    # 在测试集中的第2类上重新测一下
    #all_path = os.listdir('data/fer2013/test/2/')
    ims = []
    #for path in all_path:
        #im = Image.open('data/fer2013/test/2/'+path)
        #im = im.convert('L')
        #im = train_transform(im)
        #im = im.repeat_interleave(3, dim = 0)
        #ims.append(im.numpy())
    for im in face_ims:
    	im = im.convert('L')
    	im = train_transform(im)
    	im = im.repeat_interleave(3, dim = 0)
    	ims.append(im.numpy())
    
    ims = np.asarray(ims)
    print (ims.shape)
    ims = torch.tensor(ims)
    ims = ims.cuda()
    
    with torch.no_grad():
        score = model(ims)
    # print (score)
    y_ = torch.argmax(score,dim = 1).cpu().numpy()
    #print (y_)
    return y_



def main():
    # train()
    use_model()

    print ('End')


if __name__ == '__main__':
    main()
