import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# Configuration
image_size = 64  # DCGAN standard input size
latent_dim = 100
batch_size = 64
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class ImageOnlyDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.imgs = [f for f in sorted(os.listdir(img_dir)) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

img_dir = 'extractor/data/test/'
dataset = ImageOnlyDataset(img_dir=img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Initialize models
G = Generator().to(device)
D = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Labels
real_label = 1.0
fake_label = 0.0

# Training Loop
print("Starting GAN training...")
for epoch in range(epochs):
    for i, data in enumerate(tqdm(dataloader)):
        # Update D
        D.zero_grad()
        real_data = data.to(device)
        b_size = real_data.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        output = D(real_data).view(-1)
        lossD_real = criterion(output, label)
        lossD_real.backward()

        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake_data = G(noise)
        label.fill_(fake_label)
        output = D(fake_data.detach()).view(-1)
        lossD_fake = criterion(output, label)
        lossD_fake.backward()
        optimizerD.step()

        # Update G
        G.zero_grad()
        label.fill_(real_label)  # Generator wants D to think fake is real
        output = D(fake_data).view(-1)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

    print(f"Epoch [{epoch+1}/{epochs}] Loss_D: {lossD_real+lossD_fake:.4f} Loss_G: {lossG:.4f}")

    # Save sample generated images
    with torch.no_grad():
        fake_sample = G(torch.randn(16, latent_dim, 1, 1, device=device)).detach().cpu()
        grid = torchvision.utils.make_grid(fake_sample, padding=2, normalize=True)
        torchvision.utils.save_image(grid, f'extractor/synthetic/samples_epoch_{epoch+1}.png')

# Save generator
torch.save(G.state_dict(), 'extractor/models/generator_dcgan.pth')
print("Generator model saved.")