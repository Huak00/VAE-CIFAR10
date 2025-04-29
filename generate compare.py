import torch
import os
import matplotlib.pyplot as plt
from model import VAE


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#
latent_dim = 128
models_to_load = [100, 400]  # 你已经有100轮和200轮的模型

#
output_dir = f'outputs/compare_{latent_dim}'
os.makedirs(output_dir, exist_ok=True)

#
for epoch in models_to_load:
    #
    vae = VAE(latent_dim=latent_dim).to(device)
    model_path = f'models/vae_epoch{epoch}.pth'
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()

    #
    z = torch.randn(16, latent_dim).to(device)

    with torch.no_grad():
        generated_images = vae.decoder_fc(z).view(-1, 128, 4, 4)
        generated_images = vae.decoder(generated_images).cpu()

    #
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        img = (generated_images[i].permute(1, 2, 0).numpy() + 1) / 2  # 反归一化
        img = img.clip(0, 1)
        ax.imshow(img)
        ax.axis('off')

    plt.suptitle(f'Generated Images at Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/generated_epoch{epoch}.png')
    print(f"✅ Saved generated images at epoch {epoch} to {output_dir}/generated_epoch{epoch}.png")
    plt.close()