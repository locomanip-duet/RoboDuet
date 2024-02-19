import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义CVAE模型
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[256, 128, 64]):
        super(CVAE, self).__init__()
        # 编码器层
        self.encoder_inlayer = nn.Linear(input_dim, hidden_dims[0])
        modules = []
        for i in range(1, len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                    # nn.BatchNorm1d(hidden_dims[i]),
                    nn.Tanh())
            )
        self.encoder_hidden = nn.Sequential(*modules)
        self.encoder_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        hidden_dims.reverse()
        # 解码器层
        self.decoder_inlayer = nn.Linear(latent_dim, hidden_dims[0])
        modules = []
        for i in range(1, len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                    # nn.BatchNorm1d(hidden_dims[i]),
                    nn.Tanh())
            )
        self.decoder_hidden = nn.Sequential(*modules)
        self.decoder_outlayer = nn.Linear(hidden_dims[-1], input_dim)
    
    def encode(self, x):
        # 编码器
        out = F.tanh(self.encoder_inlayer(x))
        out = self.encoder_hidden(out)
        z_mean = self.encoder_mean(out)
        z_logvar = self.encoder_logvar(out)
        return z_mean, z_logvar
    
    def reparameterize(self, z_mean, z_logvar):
        # 重参数化技巧
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std
    
    def decode(self, z):
        # 解码器
        out = F.tanh(self.decoder_inlayer(z))
        out = self.decoder_hidden(out)
        x_recon = self.decoder_outlayer(out)
        return x_recon
    
    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
         
        z = self.reparameterize(z_mean, z_logvar)
        x_recon = self.decode(z)
        
        return x_recon, z_mean, z_logvar
    
    
    def distribution(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        return z


# 损失函数
def loss_function(x_recon, x, z_mean, z_logvar):
    
    sigma = 1
    # 重构损失
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / (sigma**2)
    # weight
    
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1)
    
    # BCE loss
    # bce_loss = F.binary_cross_entropy(valid_pred, valid_mask, reduction='sum')
    return recon_loss / x.shape[0], kl_loss.mean()
    # return recon_loss, kl_loss.sum()


# 损失函数（包括均匀分布的KL loss）
def loss_function_uniform(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # 均匀分布的KL loss计算
    uniform_min = 0.0
    uniform_max = 1.0
    kl_loss = -0.5 * torch.sum(1 + log_var - 0.5 * torch.log(torch.tensor(1.0)) - mu.pow(2) - 0.5 * log_var.exp())
    
    return BCE + kl_loss