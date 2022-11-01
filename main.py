import argparse
from invert import Inverter

# Arguments
parser = argparse.ArgumentParser(description='Invert StyleGAN')

parser.add_argument('--exp_detail', type=str, default='Invert StyleGAN')
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--seed', type=int, default=100)

parser.add_argument('--model_path', type=str, default='./pre-trained/FFHQ(pretrained).model')
parser.add_argument('--img_path', type=str, default='./example_imgs/05000.png')

parser.add_argument('--latent_type', type=str, default='mean_style')
parser.add_argument('--iterations', type=int, default=100000)
parser.add_argument('--save_iter', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--alpha', type=float, default=50)
parser.add_argument('--style_weight', type=float, default=0.7)

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=tuple, default=(0.5, 0.5, 0.5))
parser.add_argument('--std', type=tuple, default=(0.5, 0.5, 0.5))

opt = parser.parse_args()

inverter = Inverter(opt)
inverter.run()
