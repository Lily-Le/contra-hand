import os
import cv2
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from nets.ResNet import resnet50
import moco.loader
import moco.builder
import torchvision.models as models

class ModelWrap:
    def __init__(self):
        model = resnet50(pretrained=False, head_type='embed')
        # model = moco.builder.MoCo(
        #     models.__dict__["resnet50"],
        #     resnet50(False),
        #     128, 65536, 0.999, 0.07, False)
        # model = moco.builder.MoCo(
        #     models.__dict__[args.arch],
        #     args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)

        # model = nn.DataParallel(model)

        state_dict = torch.load('ckpt/model_moco.pth')
        # checkpoint = torch.load('/media/d3-ai/E/cll/Results/MoCo/logcheckpoint_0125.pth.tar')
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(state_dict)
        # model = model()
        model.cuda()
        model.eval()
        self.model = model
        self.base_path = "./rgb_variants/"

    def run2(self, image_file1,image_file2):
        img1 = cv2.imread(os.path.join(self.base_path, image_file1))
        img2 = cv2.imread(os.path.join(self.base_path, image_file2))
        print(img1.shape)
        print(img2.shape)
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))
        plt.imshow(img1)
        plt.imshow(img2)
        trafo = lambda x: np.transpose(x[:, :, ::-1], [2, 0, 1]).astype(np.float32) / 255.0 - 0.5
        img1_t = trafo(img1)
        img2_t = trafo(img2)
        print(img1_t.shape)
        batch_q = torch.Tensor(np.stack([img1_t,img1_t], 0)).cuda()
        batch_k = torch.Tensor(np.stack([img2_t,img2_t], 0)).cuda()
        # print(batch.shape)
        embed = self.model(batch_q,batch_k)
        embed = embed.detach().cpu().numpy()

        return embed

    def run(self, image_file):
        img = cv2.imread(os.path.join(self.base_path, image_file))
        print(img.shape)
        img = cv2.resize(img, (224, 224))
        plt.imshow(img)
        trafo = lambda x: np.transpose(x[:, :, ::-1], [2, 0, 1]).astype(np.float32) / 255.0 - 0.5
        img_t = trafo(img)
        print(img_t.shape)
        batch = torch.Tensor(np.stack([img_t], 0)).cuda()
        print(batch.shape)
        embed = self.model(batch)
        embed = embed.detach().cpu().numpy()

        return embed

if __name__ == '__main__':

    m = ModelWrap()
    f1 = '0007/cam4/00000016_5.jpg'
    f2 = '0007/cam4/00000017_4.jpg'
    # embed = m.run(f1), m.run(f2)
    embed = m.run2(f1, f2)
    def cossim(x, y):
        ip = np.sum(np.multiply(x, y))
        n1 = np.linalg.norm(x, 2)
        n2 = np.linalg.norm(y, 2)
        return ip / (n1*n2)

    print('score', cossim(embed[0], embed[1]))
