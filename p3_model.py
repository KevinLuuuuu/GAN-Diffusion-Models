import torch.nn as nn
import torch

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.alpha
        return grad_output, None

    def grad_reverse(x, alpha):
        return GradReverse.apply(x, alpha)

class DANN(nn.Module):
    def __init__(self, num_classes=10):
        super(DANN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 5),
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(48*4*4, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Linear(192, num_classes),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(48*4*4, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 2),
        )

    def forward(self, x, alpha):
        x = x.expand(x.data.shape[0], 3, 28, 28)
        feat = self.feature(x)
        feat = feat.view(-1, 48 * 4 * 4)

        class_out = self.classifier(feat)  

        domain_out = GradReverse.grad_reverse(feat, alpha)
        domain_out = self.discriminator(domain_out)

        return class_out, domain_out, feat