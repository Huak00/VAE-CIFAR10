import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder:
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 输出: (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 输出: (64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 输出: (128, 4, 4)
            nn.ReLU()
        )

        #
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)       #
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)    #

        # Decoder: 从 latent vector 恢复成图像
        self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4)   #
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 输出: (64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 输出: (32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),   # 输出: (3, 32, 32)
            nn.Tanh()  #
        )

    def reparameterize(self, mu, logvar):
        """

        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """

        """
        # Encode
        x = self.encoder(x)
        x = x.view(x.size(0), -1)              # 展平
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x = self.decoder_fc(z)
        x = x.view(x.size(0), 128, 4, 4)        # 恢复 feature map 形状
        recon_x = self.decoder(x)

        return recon_x, mu, logvar
