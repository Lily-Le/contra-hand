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
        model = moco.builder.MoCo(
            models.__dict__["resnet50"],
            128, 65536, 0.999, 0.07, False )
        model.cuda()
        model.eval()

        # state_dict = torch.load('ckpt/model_moco.pth')
        checkpoint = torch.load('/media/d3-ai/E/cll/Results/MoCo/logcheckpoint_0125.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])

        self.model = model
        self.base_path = "./rgb_variants/"

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
    embed = m.run(f1), m.run(f2)

    def cossim(x, y):
        ip = np.sum(np.multiply(x, y))
        n1 = np.linalg.norm(x, 2)
        n2 = np.linalg.norm(y, 2)
        return ip / (n1*n2)

    print('score', cossim(embed[0], embed[1]))
