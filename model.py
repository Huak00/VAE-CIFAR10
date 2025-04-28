import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(128*4*4, latent_dim)
        self.fc_logvar = nn.Linear(128*4*4, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 128*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

# -----------------------------
# 定义 VAE 模型
# -----------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # [B,32,16,16]
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), # [B,64,8,8]
            nn.ReLU(True),
            nn.Conv2d(64, 128,4, 2, 1), # [B,128,4,4]
            nn.ReLU(True)
        )
        self.fc_mu     = nn.Linear(128*4*4, latent_dim)
        self.fc_logvar = nn.Linear(128*4*4, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 128*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # [B,64,8,8]
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # [B,32,16,16]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3,  4, 2, 1),   # [B,3,32,32]
            nn.Tanh()  # 输出范围 [-1,1]
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(z.size(0), 128, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# -----------------------------
# 定义损失函数
# -----------------------------
def loss_function(recon_x, x, mu, logvar):
    # 重建损失（MSE）
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL 散度：D_KL(q(z|x) || p(z))
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld, recon_loss, kld

# -----------------------------
# 训练与测试流程
# -----------------------------
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for x, _ in dataloader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, recon_l, kld = loss_function(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader.dataset)

def test_epoch(model, dataloader, device, epoch, save_dir):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, _, _ = loss_function(recon, x, mu, logvar)
            total_loss += loss.item()

        # 每个 epoch 保存一些重建与随机生成的样本
        os.makedirs(save_dir, exist_ok=True)
        # 重建示例
        utils.save_image(torch.cat([x[:8], recon[:8]]),
                         f"{save_dir}/recon_epoch{epoch}.png",
                         nrow=8, normalize=True, range=(-1,1))
        # 随机生成
        z = torch.randn(64, model.fc_mu.out_features).to(device)
        sample = model.decode(z)
        utils.save_image(sample,
                         f"{save_dir}/sample_epoch{epoch}.png",
                         nrow=8, normalize=True, range=(-1,1))

    return total_loss / len(dataloader.dataset)

# -----------------------------
# 主函数
# -----------------------------
def main():
    # 超参数
    batch_size = 128
    epochs     = 30
    lr         = 1e-3
    latent_dim = 128
    save_dir   = "./results"

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据准备：CIFAR-10，像素归一化到 [-1,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    train_ds = datasets.CIFAR10("./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)

    # 模型与优化器
    model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    for epoch in range(1, epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss  = test_epoch(model, test_loader,  device, epoch, save_dir)
        print(f"Epoch {epoch:02d} | Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}")

        # 每 10 个 epoch 保存一次模型
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"./checkpoints/vae_epoch{epoch}.pth")

if __name__ == "__main__":
    main()

