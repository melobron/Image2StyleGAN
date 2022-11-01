import os
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import cv2
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models.StyleGAN2 import StyledGenerator
from lpips.loss import LPIPS
from utils import *


class Inverter:
    def __init__(self, args):
        super(Inverter, self).__init__()

        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # Random Seeds
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Parameters
        self.model_path = args.model_path
        self.img_path = args.img_path
        self.latent_type = args.latent_type

        self.lr = args.lr
        self.iterations = args.iterations
        self.save_iter = args.save_iter
        self.mean = args.mean
        self.std = args.std
        self.img_size = args.img_size
        self.step = int(math.log(self.img_size, 2)) - 2
        self.alpha = args.alpha
        self.style_weight = args.style_weight

        # Model
        self.G = StyledGenerator().to(self.device)
        ckpt = torch.load(self.model_path)
        self.G.load_state_dict(ckpt['generator'], strict=False)
        self.G.eval()

        # Mean Latent
        self.mean_style = self.get_mean_style(generator=self.G, device=self.device, style_mean_num=10)

        # Transform
        self.transform = transforms.Compose(get_transforms(args))

        # Criterion
        self.lpips_criterion = LPIPS(device=self.device, net_type='alex').to(self.device).eval()
        self.MSE_criterion = nn.MSELoss().to(self.device)

        # Directories
        self.exp_dir = make_exp_dir('./experiments/')['new_dir']
        self.exp_num = make_exp_dir('./experiments/')['new_dir_num']
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.result_path = os.path.join(self.exp_dir, 'results')

        # Tensorboard
        self.summary = SummaryWriter('runs/exp{}'.format(self.exp_num))

    def prepare(self):
        # Save Paths
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # Save Argument file
        param_file = os.path.join(self.exp_dir, 'params.json')
        with open(param_file, mode='w') as f:
            json.dump(self.args.__dict__, f, indent=4)

    def read_img(self, img_path):
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img

    def save_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(self.result_path, 'original.png'), img)

    def initial_latent(self, latent_type):
        if latent_type == 'randn':
            return torch.randn((1, 512)).to(self.device)
        elif latent_type == 'zero':
            return torch.zeros((1, 512)).to(self.device)
        elif latent_type == 'mean_style':
            return self.mean_style
        else:
            raise NotImplementedError

    @torch.no_grad()
    def get_mean_style(self, generator, device, style_mean_num):
        mean_style = None

        for _ in range(style_mean_num):
            style = generator.mean_style(torch.randn(1024, 512).to(device))
            if mean_style is None:
                mean_style = style
            else:
                mean_style += style

        mean_style /= style_mean_num
        return mean_style

    def run(self):
        self.prepare()
        self.save_img(img_path=self.img_path)

        img = self.read_img(img_path=self.img_path).to(self.device)
        latent = self.initial_latent(latent_type=self.latent_type).to(self.device)
        latent.requires_grad = True
        optimizer = optim.Adam([latent], lr=self.lr)

        for iteration in range(1, self.iterations+1):
            decoded_img = self.G.forward_from_style(style=latent, step=self.step, alpha=self.alpha,
                                                    mean_style=self.mean_style, style_weight=self.style_weight)
            lpips_loss = self.lpips_criterion(decoded_img, img)
            mse_loss = self.MSE_criterion(decoded_img, img)
            loss = lpips_loss + mse_loss
            loss.backward()
            optimizer.step()

            print('Iteration {} | total loss:{} | lpips loss:{}, mse loss:{}'.format(
                iteration, loss.item(), lpips_loss.item(), mse_loss.item()
            ))

            if iteration % self.save_iter == 0 or iteration == self.iterations:
                reverse_transform = transforms.Compose([
                    transforms.Normalize(mean=[-m / s for m, s in zip(self.mean, self.std)], std=[1 / s for s in self.std])
                ])
                sample = torch.squeeze(decoded_img, dim=0)
                sample = reverse_transform(sample)
                sample = sample.detach().cpu().numpy().transpose(1, 2, 0)
                sample = np.clip(sample, 0., 1.) * 255.
                sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.result_path, '{}iterations.png'.format(iteration)), sample)
