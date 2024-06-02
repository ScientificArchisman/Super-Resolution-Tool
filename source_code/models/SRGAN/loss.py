from torchvision import models
import torch.nn as nn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device).eval()
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:35]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.feature_extractor(x)
    
class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.vgg = VGG19FeatureExtractor()
        self.mse_loss = nn.MSELoss()

    def forward(self, sr, hr):
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        content_loss = self.mse_loss(sr_features, hr_features)
        return content_loss

        

class GeneratoradversarialLoss(nn.Module):
    """Generator adversarial loss function.
    Args:
        fake_preds (torch.Tensor): Fake image predictions from the discriminator."""
    def __init__(self):
        super(GeneratoradversarialLoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, fake_preds):
        target = torch.ones_like(fake_preds)
        return self.criterion(fake_preds, target)


class DiscriminatorLoss(nn.Module):
    """Discriminator loss function.
    Args:
        real_preds (torch.Tensor): Real image predictions from the discriminator.
        fake_preds (torch.Tensor): Fake image predictions from the discriminator."""
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, real_preds, fake_preds):
        real_targets = torch.ones_like(real_preds)
        fake_targets = torch.zeros_like(fake_preds)

        real_loss = self.criterion(real_preds, real_targets)
        fake_loss = self.criterion(fake_preds, fake_targets)

        return (real_loss + fake_loss) / 2
    



