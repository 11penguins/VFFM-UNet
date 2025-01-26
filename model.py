import torch
import torch.nn as nn
from torch.amp import autocast
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_

class FastSelfAttention(nn.Module):
    def __init__(self, dim, num_head, qkv_bias=False, attention_drop=0.1, proj_drop=0.1):
        super().__init__()

        self.dim1 = dim
        self.num_heads = num_head
        head_dim = dim // num_head

        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, self.num_heads, bias=qkv_bias)
        self.k = nn.Linear(dim, self.num_heads, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attention_drop = nn.Dropout(attention_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.conv2d = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim)
        self.activate = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, H * W, C)
        B, N, C = x.shape

        assert N == H * W

        q = self.q(x).reshape(B, N, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        alpha = self.softmax(q)
        q = alpha * q
        q = torch.sum(q, dim=1, keepdim=True)

        x_new = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_new = self.conv2d(x_new).reshape(B, C, -1).permute(0, 2, 1)
        x_new = self.norm(x_new)
        x_activated = self.activate(x_new)

        k = self.k(x_activated).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        p = q * k
        belta = self.softmax(p)

        v = self.v(x_activated).reshape(B, -1, C, 1).permute(0, 2, 1, 3)

        pool1 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        belta = pool1(belta)

        pool2 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        p = pool2(p)
        t = belta @ p.transpose(-2, -1)
        _, _, h, w = t.shape
        t = F.interpolate(t, size=(2 * h, 2 * w), mode='bilinear', align_corners=False)

        attention = t * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attention_drop(attention)
        attention = torch.sum(attention, dim=1, keepdim=True)
        x = (attention @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.reshape(B, C, H, W)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_path=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.activate = nn.GELU()

        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop_path)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activate(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CS(nn.Module):
    def __init__(self, dim):
        super(CS, self).__init__()
        self.fc1 = nn.Linear(dim, dim//2)
        self.fc2 = nn.Linear(dim//2, dim)
        self.bn = nn.BatchNorm1d(dim//2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = x1 + x2

        avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        SC = avg_pool(x)
        SC = SC.contiguous().view(SC.size(0), -1)

        Zc = self.fc1(SC)
        Zc = self.bn(Zc)
        Zc = self.relu(Zc)

        Z = self.fc2(Zc)

        Z = self.sigmoid(Z).unsqueeze(2).unsqueeze(3)
        belta = Z
        alpha = 1 - belta

        ans = belta * x1 + alpha * x2
        return ans

class Granularity_Fusion(nn.Module):
    def __init__(self, dim, num_head, image_size):
        super().__init__()
        self.dim = dim
        self.size = image_size
        self.conv_lk1 = FastSelfAttention(dim=dim, num_head=num_head)
        self.conv_lk2 = FastSelfAttention(dim=dim, num_head=num_head)
        self.conv_lk3 = FastSelfAttention(dim=dim, num_head=num_head)
        self.conv0 = nn.Conv2d(self.dim, self.dim // 3, 1)
        self.conv1 = nn.Conv2d(2, 3, 7, padding=3)
        self.conv2 = nn.Conv2d(self.dim // 3, self.dim, 1)

        self.cs = CS(dim=self.dim)

    def forward(self, x):
        attn1 = self.conv_lk1(x)
        pool = nn.AvgPool2d(2, stride=2)
        unpool = nn.Upsample(scale_factor=2, mode='nearest')
        attn2 = pool(attn1)
        attn2 = self.conv_lk2(attn2)
        t = attn2
        attn2 = unpool(attn2)
        attn3 = pool(t)
        attn3 = self.conv_lk3(attn3)
        unpool = nn.Upsample(scale_factor=4, mode='nearest')
        attn3 = unpool(attn3)

        feature1 = attn2
        feature2 = attn3
        ans = self.cs(feature1, feature2)


        U_1 = self.conv0(attn1)
        U_2 = self.conv0(attn2)
        U_3 = self.conv0(attn3)

        U = torch.cat([U_1, U_2, U_3], dim=1)
        avg_U = torch.mean(U, dim=1, keepdim=True)
        max_U, _ = torch.max(U, dim=1, keepdim=True)
        SA = torch.cat([avg_U, max_U], dim=1)
        mask = self.conv1(SA).sigmoid()
        attn = U_1 * mask[:, 0, :, :].unsqueeze(1) + U_2 * mask[:, 1, :, :].unsqueeze(1) + U_3 * mask[:, 2, :, :].unsqueeze(1)
        S = self.conv2(attn)
        S = S * x

        return S + ans

class Fastformer(nn.Module):
    def __init__(self, dim, num_head, image_size):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.attention = Granularity_Fusion(dim=dim, num_head=num_head, image_size=image_size)
        self.mlp = MLP(in_features=dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = x + self.attention(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = x + self.mlp(x)
        return x

class Net(nn.Module):
    def __init__(self, img_size, head_num, depth=[8, 16, 24, 32, 48, 64], channel=3):
        super().__init__()
        self.gelu = nn.GELU()
        self.img_size = img_size
        self.norm0 = nn.GroupNorm(4, depth[0])
        self.norm1 = nn.GroupNorm(4, depth[1])
        self.norm2 = nn.GroupNorm(4, depth[2])
        self.norm3 = nn.GroupNorm(4, depth[3])
        self.norm4 = nn.GroupNorm(4, depth[4])
        self.norm5 = nn.GroupNorm(4, depth[5])

        self.encoder0 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=depth[0], kernel_size=3, padding=1, stride=1)
        )
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=depth[0], out_channels=depth[1], kernel_size=3, padding=1, stride=1)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=depth[1], out_channels=depth[2], kernel_size=3, padding=1, stride=1)
        )
        self.encoder3 = nn.Sequential(
            Fastformer(dim=depth[2], num_head=head_num, image_size=self.img_size//8),
            nn.Conv2d(in_channels=depth[2], out_channels=depth[3], kernel_size=3, padding=1, stride=1)
        )
        self.encoder4 = nn.Sequential(
            Fastformer(dim=depth[3], num_head=head_num, image_size=self.img_size//16),
            nn.Conv2d(in_channels=depth[3], out_channels=depth[4], kernel_size=3, padding=1, stride=1)
        )
        self.encoder5 = nn.Sequential(
            Fastformer(dim=depth[4], num_head=head_num, image_size=self.img_size//32),
            nn.Conv2d(in_channels=depth[4], out_channels=depth[5], kernel_size=3, padding=1, stride=1),
            Fastformer(dim=depth[5], num_head=head_num, image_size=self.img_size//32)
        )

        self.decoder0 = nn.Sequential(
            nn.Conv2d(in_channels=depth[5], out_channels=depth[4], kernel_size=3, stride=1, padding=1),
            Fastformer(dim=depth[4], num_head=head_num, image_size=self.img_size//16)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_channels=depth[4], out_channels=depth[3], kernel_size=3, stride=1, padding=1),
            Fastformer(dim=depth[3], num_head=head_num, image_size=self.img_size//8)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(in_channels=depth[3], out_channels=depth[2], kernel_size=3, stride=1, padding=1),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(in_channels=depth[2], out_channels=depth[1], kernel_size=3, stride=1, padding=1)
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(in_channels=depth[1], out_channels=depth[0], kernel_size=3, stride=1, padding=1)
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(in_channels=depth[0], out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        self.connection1 = nn.Conv2d(in_channels=2*depth[4], out_channels=depth[4], kernel_size=3, stride=1, padding=1)
        self.connection2 = nn.Conv2d(in_channels=2*depth[3], out_channels=depth[3], kernel_size=3, stride=1, padding=1)
        self.connection3 = nn.Conv2d(in_channels=2*depth[2], out_channels=depth[2], kernel_size=3, stride=1, padding=1)
        self.connection4 = nn.Conv2d(in_channels=2*depth[1], out_channels=depth[1], kernel_size=3, stride=1, padding=1)
        self.connection5 = nn.Conv2d(in_channels=2*depth[0], out_channels=depth[0], kernel_size=3, stride=1, padding=1)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        out = self.gelu(self.norm0(self.encoder0(x)))
        out_encoder_0 = out
        out = F.avg_pool2d(out, kernel_size=2, stride=2)

        out = self.gelu(self.norm1(self.encoder1(out)))
        out_encoder_1 = out
        out = F.avg_pool2d(out, kernel_size=2, stride=2)

        out = self.gelu(self.norm2(self.encoder2(out)))
        out_encoder_2 = out
        out = F.avg_pool2d(out, kernel_size=2, stride=2)

        out = self.gelu(self.norm3(self.encoder3(out)))
        out_encoder_3 = out
        out = F.avg_pool2d(out, kernel_size=2, stride=2)

        out = self.gelu(self.norm4(self.encoder4(out)))
        out_encoder_4 = out
        out = F.avg_pool2d(out, kernel_size=2, stride=2)

        out = self.gelu(self.norm5(self.encoder5(out)))

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.gelu(self.norm4(self.decoder0(out)))
        out = torch.cat((out, out_encoder_4), dim=1)
        out = self.connection1(out)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.gelu(self.norm3(self.decoder1(out)))
        out = torch.cat((out, out_encoder_3), dim=1)
        out = self.connection2(out)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.gelu(self.norm2(self.decoder2(out)))
        out = torch.cat((out, out_encoder_2), dim=1)
        out = self.connection3(out)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.gelu(self.norm1(self.decoder3(out)))
        out = torch.cat((out, out_encoder_1), dim=1)
        out = self.connection4(out)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.gelu(self.norm0(self.decoder4(out)))
        out = torch.cat((out, out_encoder_0), dim=1)
        out = self.connection5(out)

        out = self.decoder5(out)

        return torch.sigmoid(out)