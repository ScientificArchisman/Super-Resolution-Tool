from torchvision.models import vgg19
import torch.nn as nn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = vgg19(pretrained=True).features[:36].to(device).eval()
        self.loss = nn.MSELoss()

    def forward(self, sr, hr):
        sr = self.vgg(sr)
        hr = self.vgg(hr)
        return self.loss(sr, hr)
    

class GeneratoradversarialLoss(nn.Module):
    def __init__(self):
        super(GeneratoradversarialLoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, fake_preds):
        target = torch.ones_like(fake_preds)
        return self.criterion(fake_preds, target)


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, real_preds, fake_preds):
        real_targets = torch.ones_like(real_preds)
        fake_targets = torch.zeros_like(fake_preds)

        real_loss = self.criterion(real_preds, real_targets)
        fake_loss = self.criterion(fake_preds, fake_targets)

        return (real_loss + fake_loss) / 2
    



