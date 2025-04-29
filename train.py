import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import VAE
import os
import matplotlib.pyplot as plt

# -------------------------
# 1. 设置设备
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# -------------------------
# 2. 超参数设置
# -------------------------
latent_dim = 128
start_epoch = 1          # 从第50轮开始
total_epochs = 200       # 总共训练到400轮
batch_size = 128
learning_rate = 0.001
save_every = 100          # 每100轮保存一次

# -------------------------
# 3. 数据加载
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# -------------------------
# 4. 加载之前训练到50轮的模型
# -------------------------
vae = VAE(latent_dim=latent_dim).to(device)

vae.train()


# -------------------------
# 5. 设置优化器
# -------------------------
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# -------------------------
# 6. 定义损失函数
# -------------------------
def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# -------------------------
# 7. 开始继续训练
# -------------------------
losses = []

for epoch in range(start_epoch, total_epochs):
    running_loss = 0
    vae.train()

    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)

        recon_images, mu, logvar = vae(images)
        loss = vae_loss_function(recon_images, images, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader.dataset)
    losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{total_epochs}], Loss: {avg_loss:.4f}')

    # 每100轮保存一次模型
    if (epoch+1) % save_every == 0:
        os.makedirs('models', exist_ok=True)
        torch.save(vae.state_dict(), f'models/vae_epoch{epoch+1}.pth')
        print(f'✅ Saved model at models/vae_epoch{epoch+1}.pth')

# -------------------------
# 8. 保存最终模型
# -------------------------
torch.save(vae.state_dict(), 'models/vae_final.pth')
print('✅ Final model saved at models/vae_final.pth')

# -------------------------
# 9. 绘制训练损失曲线
# -------------------------
os.makedirs('outputs', exist_ok=True)
plt.figure()
plt.plot(range(start_epoch+1, total_epochs+1), losses, marker='o')
plt.title('VAE Training Loss (Continue)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('outputs/training_loss_continue.png')
plt.show()
print('✅ Training loss curve saved at outputs/training_loss_continue.png')

