from src import detect_faces, show_bboxes
from PIL import Image, ImageDraw, ImageFont
import glob
import os
import pandas as pd
import imageio

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

import sys
sys.path.append('FER/')
import main

# 1.detect face
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,keep_all=True, #return all faces 
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 2.detact identity
# import identity embedding
feature_embedding = torch.load('../IdentityRecognition/embeddings.pt')
feature_embedding = feature_embedding.to(device)
names = ['Chatur', 'Dean', 'Farhan', 'Joy', 'Raju', 'Rancho']


# 3.run the main loop and save the results of images
def detect_frame(img):
    fontStyle = ImageFont.truetype("LiberationSans-Regular.ttf", 25,encoding="utf-8")
    # 1.face recognition
    faces = mtcnn(img)  # 直接infer所有的faces
    #但是这里相当于两次infer，会浪费时间
    boxes, _ = mtcnn.detect(img)  # 检测出人脸框 返回的是位置
    frame_draw = img.copy()
    draw = ImageDraw.Draw(frame_draw)
    #print("检测人脸数目：",len(boxes))
    if boxes is not None:
        index_len = len(boxes)
        emotion_identity = pd.DataFrame(columns = ['name','emotion'],index = range(index_len))
        for i,box in enumerate(boxes):
            face_embedding = resnet(faces[i].unsqueeze(0).to(device))
            #print(face_embedding.size(),'大小')
            # 计算距离
            probs = [(face_embedding - feature_embedding[i]).norm().item() for i in range(feature_embedding.size()[0])]
            #print(probs)
            # 我们可以认为距离最近的那个就是最有可能的人，但也有可能出问题，数据库中可以存放一个人的多视角多姿态数据，对比的时候可以采用其他方法，如投票机制决定最后的识别人脸

            # 2.identity recognition
            index = probs.index(min(probs))   # 对应的索引就是判断的人脸
            name = names[index] # 对应的人脸
            if probs[probs.index(min(probs))] < 1:
                draw.rectangle(box.tolist(), outline='red',width=5)  # 绘制框
                draw.text((int(box[0]),int(box[1])+70), str(name), fill=(255,0,0),font=fontStyle)

                # 3.emotion recognition
                # detect emotion
                face = [img.crop(box)]
                emotion = [main.use_model(im) for im in face]
                emotion = emotion[0][0]
                if emotion == 0:
                    emotion = 'anger'
                elif emotion == 1:
                    emotion = 'disgust'
                elif emotion == 2:
                    emotion = 'fear'
                elif emotion == 3:
                    emotion = 'happy'
                elif emotion == 4:
                    emotion = 'sad'
                elif emotion == 5:
                    emotion = 'surprised'
                elif emotion == 6:
                    emotion = 'neutral'
                draw.text((int(box[0]),int(box[1])-70), str(emotion), fill='yellow',font=fontStyle)
                emotion_identity.iloc[i,:] = [name,emotion]
    else:
        emotion_identity = pd.DataFrame(columns = ['name','emotion'],index = range(0))
            
    return frame_draw,emotion_identity

