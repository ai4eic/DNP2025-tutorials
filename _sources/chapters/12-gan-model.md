# GAN for FCAL Showers

## Introduction

Generative Adversarial Networks (GANs) can generate realistic FCAL shower images, providing a fast alternative to traditional Monte Carlo simulations. In this chapter, we'll build GANs to generate showers for photons, π⁺, and π⁻ particles.

## GAN Fundamentals

### The Two-Player Game

GANs consist of two neural networks competing against each other:

**Generator (G)**:
- Input: Random noise vector z (latent space)
- Output: Fake FCAL shower image
- Goal: Generate realistic showers that fool the discriminator

**Discriminator (D)**:
- Input: FCAL shower image (real or fake)
- Output: Probability that input is real
- Goal: Correctly distinguish real from fake showers

### Training Objective

**Discriminator Loss**:
```
L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
```
Maximize probability of correct classification

**Generator Loss**:
```
L_G = -E[log D(G(z))]
```
Minimize discriminator's ability to detect fakes (equivalently: maximize log probability that fakes are classified as real)

### Training Algorithm

```
for epoch in epochs:
    for batch in dataloader:
        # Train Discriminator
        1. Get real images from dataset
        2. Generate fake images from noise
        3. Compute D loss on real and fake
        4. Update D parameters
        
        # Train Generator
        5. Generate fake images from new noise
        6. Compute G loss (fool discriminator)
        7. Update G parameters
```

## Vanilla GAN Architecture

### Generator

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, image_size=32, num_channels=1):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Calculate initial size after upsampling
        self.init_size = image_size // 4  # 8 for 32x32 images
        self.fc = nn.Linear(latent_dim, 128 * self.init_size * self.init_size)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            
            # Upsample to image_size // 2
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Upsample to image_size
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final conv to get correct number of channels
            nn.Conv2d(64, num_channels, 3, stride=1, padding=1),
            nn.ReLU()  # Energy must be positive
        )
    
    def forward(self, z):
        # Project and reshape
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        
        # Generate image
        img = self.conv_blocks(out)
        return img
```

### Discriminator

```python
class Discriminator(nn.Module):
    def __init__(self, image_size=32, num_channels=1):
        super(Discriminator, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            # Input: num_channels x image_size x image_size
            nn.Conv2d(num_channels, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 64 x image_size/2 x image_size/2
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 128 x image_size/4 x image_size/4
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        
        # Calculate size after conv blocks
        ds_size = image_size // 8
        self.fc = nn.Sequential(
            nn.Linear(256 * ds_size * ds_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.size(0), -1)
        validity = self.fc(out)
        return validity
```

### Training Loop

```python
import torch.optim as optim

# Initialize models
latent_dim = 100
generator = Generator(latent_dim=latent_dim, image_size=32, num_channels=1)
discriminator = Discriminator(image_size=32, num_channels=1)

# Loss function
adversarial_loss = nn.BCELoss()

# Optimizers
lr = 0.0002
beta1 = 0.5
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = generator.to(device)
discriminator = discriminator.to(device)
adversarial_loss = adversarial_loss.to(device)

# Training
num_epochs = 200
for epoch in range(num_epochs):
    for i, real_imgs in enumerate(dataloader):
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        
        # Labels for real and fake images
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Loss on real images
        real_pred = discriminator(real_imgs)
        d_loss_real = adversarial_loss(real_pred, real_labels)
        
        # Generate fake images
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(z)
        
        # Loss on fake images
        fake_pred = discriminator(fake_imgs.detach())
        d_loss_fake = adversarial_loss(fake_pred, fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        
        # Generate fake images
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(z)
        
        # Generator tries to fool discriminator
        fake_pred = discriminator(fake_imgs)
        g_loss = adversarial_loss(fake_pred, real_labels)
        
        g_loss.backward()
        optimizer_G.step()
        
        # Print progress
        if i % 100 == 0:
            print(f'[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] '
                  f'[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]')
    
    # Save generated samples
    if epoch % 10 == 0:
        save_samples(generator, epoch, latent_dim, device)
```

## DCGAN: Deep Convolutional GAN

DCGAN introduces architectural guidelines for stable GAN training:

### Guidelines

1. Replace pooling with strided convolutions (discriminator) and fractional-strided convolutions (generator)
2. Use batch normalization in both networks (except generator output and discriminator input)
3. Remove fully connected hidden layers
4. Use ReLU in generator (except output: Tanh)
5. Use LeakyReLU in discriminator

### DCGAN Generator

```python
class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100, image_size=64, num_channels=1, ngf=64):
        super(DCGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # State: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # State: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # State: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # State: ngf x 32 x 32
            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
            nn.ReLU()  # Positive energy values
            
            # Output: num_channels x 64 x 64
        )
    
    def forward(self, z):
        # Reshape z to 4D tensor
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.main(z)
```

### DCGAN Discriminator

```python
class DCDiscriminator(nn.Module):
    def __init__(self, image_size=64, num_channels=1, ndf=64):
        super(DCDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: num_channels x 64 x 64
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.main(img).view(-1, 1)
```

## Conditional GAN (cGAN)

Conditional GANs allow control over generated content by conditioning on labels (particle type).

### Architecture Modifications

**Generator**: Takes latent vector z + condition c

```python
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=3, image_size=32, num_channels=1):
        super(ConditionalGenerator, self).__init__()
        
        # Embedding for condition
        self.label_embed = nn.Embedding(num_classes, num_classes)
        
        self.latent_dim = latent_dim
        self.init_size = image_size // 4
        
        # Combined input: z + embedded label
        self.fc = nn.Linear(latent_dim + num_classes, 128 * self.init_size * self.init_size)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, num_channels, 3, stride=1, padding=1),
            nn.ReLU()
        )
    
    def forward(self, z, labels):
        # Embed labels
        label_input = self.label_embed(labels)
        
        # Concatenate z and label
        gen_input = torch.cat([z, label_input], dim=1)
        
        # Generate
        out = self.fc(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
```

**Discriminator**: Takes image + condition

```python
class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=3, image_size=32, num_channels=1):
        super(ConditionalDiscriminator, self).__init__()
        
        # Embedding for label
        self.label_embed = nn.Embedding(num_classes, image_size * image_size)
        
        # Discriminator takes image + label channel
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(num_channels + 1, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        
        ds_size = image_size // 8
        self.fc = nn.Sequential(
            nn.Linear(256 * ds_size * ds_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        # Embed and reshape labels to image size
        label_input = self.label_embed(labels)
        label_input = label_input.view(labels.size(0), 1, img.size(2), img.size(3))
        
        # Concatenate image with label channel
        d_input = torch.cat([img, label_input], dim=1)
        
        out = self.conv_blocks(d_input)
        out = out.view(out.size(0), -1)
        validity = self.fc(out)
        return validity
```

### Training Conditional GAN

```python
# Initialize
generator = ConditionalGenerator(latent_dim=100, num_classes=3)
discriminator = ConditionalDiscriminator(num_classes=3)

# Move to device
generator = generator.to(device)
discriminator = discriminator.to(device)

# Training loop
for epoch in range(num_epochs):
    for i, (real_imgs, labels) in enumerate(dataloader):
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        labels = labels.to(device)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        real_pred = discriminator(real_imgs, labels)
        d_loss_real = adversarial_loss(real_pred, real_labels)
        
        # Generate fake images with same labels
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(z, labels)
        fake_pred = discriminator(fake_imgs.detach(), labels)
        d_loss_fake = adversarial_loss(fake_pred, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(z, labels)
        fake_pred = discriminator(fake_imgs, labels)
        g_loss = adversarial_loss(fake_pred, real_labels)
        
        g_loss.backward()
        optimizer_G.step()

# Generate specific particle type
def generate_particle_shower(generator, particle_type, num_samples=16):
    """
    particle_type: 0=photon, 1=pi+, 2=pi-
    """
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        labels = torch.full((num_samples,), particle_type, dtype=torch.long, device=device)
        fake_imgs = generator(z, labels)
    return fake_imgs
```

## Physics-Conditioned GAN

For more control, condition on kinematic variables (energy, angle, etc.):

```python
class PhysicsConditionedGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_kinematic=4, image_size=32):
        """
        num_kinematic: number of physics parameters (e.g., E, theta, phi, PID)
        """
        super(PhysicsConditionedGenerator, self).__init__()
        
        # MLP for physics conditions
        self.condition_mlp = nn.Sequential(
            nn.Linear(num_kinematic, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2)
        )
        
        self.latent_dim = latent_dim
        self.init_size = image_size // 4
        
        # Combine latent and physics features
        self.fc = nn.Linear(latent_dim + 128, 128 * self.init_size * self.init_size)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.ReLU()
        )
    
    def forward(self, z, physics_params):
        # Process physics parameters
        cond_features = self.condition_mlp(physics_params)
        
        # Combine with latent vector
        gen_input = torch.cat([z, cond_features], dim=1)
        
        # Generate image
        out = self.fc(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Usage
# physics_params = [energy, theta, phi, particle_id] normalized to [0, 1]
physics_params = torch.tensor([[5.0, 0.1, 1.5, 0]], device=device)  # 5 GeV photon
z = torch.randn(1, latent_dim, device=device)
generated_shower = generator(z, physics_params)
```

## Training Tips

### Stabilizing GAN Training

**1. Use Label Smoothing**
```python
# Instead of hard 0/1 labels
real_labels = torch.ones(batch_size, 1) * 0.9  # Smooth to 0.9
fake_labels = torch.ones(batch_size, 1) * 0.1  # Smooth to 0.1
```

**2. Add Noise to Inputs**
```python
# Add small noise to discriminator inputs
noise = torch.randn_like(real_imgs) * 0.1
real_imgs_noisy = real_imgs + noise
```

**3. Balance D and G Training**
```python
# Train D more frequently if G is winning
if d_loss < g_loss:
    # Train D multiple times
    for _ in range(2):
        train_discriminator_step()
```

**4. Monitor Mode Collapse**
- Check diversity of generated samples
- Visualize latent space interpolations
- Track inception score or FID

### Evaluation Metrics

**Inception Score (IS)**:
Measures quality and diversity of generated images

**Fréchet Inception Distance (FID)**:
Measures similarity between real and generated distributions (lower is better)

```python
from pytorch_fid import fid_score

# Calculate FID
fid_value = fid_score.calculate_fid_given_paths(
    [real_data_path, fake_data_path],
    batch_size=50,
    device=device,
    dims=2048
)
print(f'FID: {fid_value:.2f}')
```

**Physics Metrics**:
- Energy distribution comparison
- Shower shape moments (RMS, skewness, kurtosis)
- Cluster multiplicity
- Spatial correlations

## Practical Example: Three-Particle GAN

```python
# Setup for γ, π+, π- generation
num_classes = 3  # 0: photon, 1: pi+, 2: pi-
particle_names = ['photon', 'pi_plus', 'pi_minus']

generator = ConditionalGenerator(latent_dim=100, num_classes=num_classes)
discriminator = ConditionalDiscriminator(num_classes=num_classes)

# Train (code from above)

# Generate showers for each particle type
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 4, figsize=(12, 9))

for particle_id, particle_name in enumerate(particle_names):
    samples = generate_particle_shower(generator, particle_id, num_samples=4)
    
    for i in range(4):
        ax = axes[particle_id, i]
        ax.imshow(samples[i, 0].cpu().numpy(), cmap='hot')
        ax.axis('off')
        if i == 0:
            ax.set_ylabel(particle_name, fontsize=12)

plt.tight_layout()
plt.savefig('generated_showers.png')
plt.show()
```

## Summary

Key points:
1. GANs learn to generate realistic FCAL showers through adversarial training
2. DCGAN architecture provides stable training
3. Conditional GANs enable particle-type-specific generation
4. Physics conditioning allows fine-grained control
5. Careful training and evaluation ensures quality generation

## Next Steps

- [VAE for FCAL Showers](13-vae-model.md) - Alternative generative approach
- [Conditional Generation](14-conditional-generation.md) - Advanced conditioning techniques
- [Physics-Informed Generation](15-physics-informed.md) - Incorporating physics constraints

## Further Reading

- Goodfellow et al., "Generative Adversarial Networks" (2014)
- Radford et al., "Unsupervised Representation Learning with Deep Convolutional GANs" (2015)
- Mirza & Osindero, "Conditional Generative Adversarial Nets" (2014)
- Paganini et al., "CaloGAN: Simulating 3D High Energy Particle Showers in Multi-Layer Electromagnetic Calorimeters with Generative Adversarial Networks" (arXiv:1705.02355)
