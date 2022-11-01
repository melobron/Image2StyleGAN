import sys
sys.path.append('.')
import torch
import torch.nn as nn

from lpips.networks import get_network, LinLayers
from lpips.utils import get_state_dict


class LPIPS(nn.Module):
    def __init__(self, device, net_type='alex', version='0.1'):
        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type).to(device)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list).to(device)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Note that the scale of input should be [-1,1]
        """
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0)) / x.shape[0]


# if __name__ == '__main__':
    # lpips = LPIPS(device=)
