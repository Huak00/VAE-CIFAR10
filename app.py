import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import VAE
import os


st.set_page_config(page_title="VAE CIFAR-10 Image Generator", layout="centered")
st.title("🖼️ Variational Autoencoder (VAE) Image Generator")
st.write("This web app uses a trained VAE model to generate or reconstruct images based on CIFAR-10 dataset.")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

latent_dim = 128
vae = VAE(latent_dim=latent_dim).to(device)


model_path = 'models/vae_final.pth'
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please train and save the model first.")
    st.stop()

vae.load_state_dict(torch.load(model_path, map_location=device))
vae.eval()

st.success('✅ Trained VAE model loaded successfully!')


st.header("Upload an Image for Reconstruction")

uploaded_file = st.file_uploader("Choose an image (jpg/png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)


    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    input_image = preprocess(image).unsqueeze(0).to(device)


    with torch.no_grad():
        recon_image, mu, logvar = vae(input_image)


    recon_image = (recon_image.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
    st.subheader("🔄 Reconstructed Image:")
    fig, ax = plt.subplots()
    ax.imshow(recon_image)
    ax.axis('off')
    st.pyplot(fig)


st.header("Or Generate New Random Images")

if st.button('🎲 Generate Random Images'):
    with torch.no_grad():
        z = torch.randn(16, latent_dim).to(device)
        generated_images = vae.decoder_fc(z).view(-1, 128, 4, 4)
        generated_images = vae.decoder(generated_images).cpu()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        img = (generated_images[i].permute(1, 2, 0).detach().numpy() + 1) / 2
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
