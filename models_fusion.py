import torch
import torch.nn as nn


class RawDepthAttention(nn.Module):
    """
    Build EEG Depth Attention module.
    :arg
    C: num of channels
    W: num of time samples
    k: learnable kernel size
    """
    def __init__(self, W, C, k=7):
        super(RawDepthAttention, self).__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)  # original kernel k
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        """
        :arg
        """
        x_pool = self.adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = self.conv(x_transpose)
        y = self.softmax(y)
        y = y.transpose(-2, -3)
        return y * self.C * x



class ConvGitShallowLMDA6(nn.Module):

    def __init__(self, img_weight=0.02, width=224, length=224, num_classes=2, samples=1000, channels=3, avepool=25):
        super(ConvGitShallowLMDA6, self).__init__()
        self.channel_weight = nn.Parameter(torch.randn(9, 1, channels), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.num_classes = num_classes
        self.img_weight = img_weight

        self.raw_time_conv = nn.Sequential(
            nn.Conv2d(9, 24, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, kernel_size=(1, 75), groups=24, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )

        self.raw_chanel_conv = nn.Sequential(
            nn.Conv2d(24, 9, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 9, kernel_size=(channels, 1), groups=9, bias=False),
            nn.BatchNorm2d(9),
            nn.GELU(),
        )

        self.raw_norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            nn.Dropout(p=0.65),
        )

        # raw features
        raw_eeg = torch.ones((1, 1, channels, samples))
        raw_eeg = torch.einsum('bdcw, hdc->bhcw', raw_eeg, self.channel_weight)
        out_raw_eeg = self.raw_time_conv(raw_eeg)

        N, C, H, W = out_raw_eeg.size()
        self.depthAttention = RawDepthAttention(W, C, k=7)

        out_raw_eeg = self.raw_chanel_conv(out_raw_eeg)
        out_raw_eeg = self.raw_norm(out_raw_eeg)
        out_raw_eeg_shape = out_raw_eeg.cpu().data.numpy().shape
        print('out_raw_eeg_shape: ', out_raw_eeg_shape)
        n_out_raw_eeg = out_raw_eeg_shape[-1] * out_raw_eeg_shape[-2] * out_raw_eeg_shape[-3]

        self.frequency_features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(4, 4), stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8),
            nn.Dropout(p=0.25),

            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3),
            nn.Dropout(p=0.25),

            nn.Conv2d(32, out_raw_eeg_shape[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_raw_eeg_shape[-1]),
            nn.Conv2d(out_raw_eeg_shape[-1], out_raw_eeg_shape[-1], kernel_size=4,
                      groups=out_raw_eeg_shape[-1], bias=False, padding=2),
            nn.BatchNorm2d(out_raw_eeg_shape[-1]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3),
            nn.Dropout(p=0.25),
        )

 
        img_eeg = torch.ones((1, 3, width, length))
        out_img = self.frequency_features(img_eeg)
        out_img_shape = out_img.cpu().data.numpy().shape
        n_out_img = out_img_shape[-1] * out_img_shape[-2] * out_img_shape[-3]
        print('n_out_img shape: ', out_img_shape)

        self.classifier = nn.Sequential(
            nn.Linear(n_out_img, num_classes),
        )


    def forward(self, x_raw, x_frequency):
        # features for frequency graph
        x_frequency = self.frequency_features(x_frequency)
        x_frequency = x_frequency.view(x_frequency.size(0), -1)

        x_raw = torch.einsum('bdcw, hdc->bhcw', x_raw, self.channel_weight)  # 导联权重筛选
        x_raw = self.raw_time_conv(x_raw)
        x_raw = self.raw_chanel_conv(x_raw)
        x_raw = self.depthAttention(x_raw)
        x_raw = self.raw_norm(x_raw)
        # raw features and img features weighted fusion

        x_raw_flatten = x_raw.view(x_raw.size(0), -1)

        weighted_features = x_raw_flatten * (1-self.img_weight) + x_frequency * self.img_weight

        x = self.classifier(weighted_features)

        return x,  x_raw_flatten, x_frequency

