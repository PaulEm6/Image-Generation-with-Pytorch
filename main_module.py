# main_module.py
import torch
import os
import random
import time
from discriminator_generator_module import DiscriminatorGeneratorModule
from data_loader_module import DataLoaderModule
from data_loader_module import DeviceDataLoader
from training_module import TrainingModule
from utils_module import UtilsModule

# Set random seed for reproducibility
manualSeed = 500
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)  # Needed for reproducible results

Path = "pokemon_jpg"
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
batch_size = 64
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # Normalization
latent_size = 100

# Record the start time
start_time = time.time()

# Create input data and configuring device
data_module = DataLoaderModule()
train_ds = data_module.create_datastorage(Path, image_size, stats)
train_dl = data_module.create_dataloader(train_ds, batch_size, shuffle=True)
device = data_module.get_default_device()

print(f'\nDevice being used is: {device}')
train_dl = DeviceDataLoader(device=device, dl=train_dl)

# Create discriminator and generator
disc_gen_module = DiscriminatorGeneratorModule()
discriminator = disc_gen_module.create_discriminator(nc, ndf)
discriminator = UtilsModule.to_device(discriminator, device)

generator = disc_gen_module.create_generator(nz, ngf, nc)
xb = torch.randn(batch_size, nz, 1, 1)  # random latent tensors
fake_images = generator(xb)
print("\nShape of a batch of images " + str(fake_images.shape) + "\n")
generator = UtilsModule.to_device(generator, device)

# Creating folder for results
sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)

# Fixed input vector for evolution of GAN
fixed_latent = torch.randn(batch_size, nz, 1, 1, device=device)
UtilsModule.save_samples(0, generator, fixed_latent,  sample_dir=sample_dir)

# Training of GAN
lr = 0.0002
epochs = 4
history = TrainingModule.fit(generator, discriminator, train_dl, fixed_latent, device, epochs, lr, batch_size=batch_size, latent_size=latent_size)
losses_g, losses_d, real_scores, fake_scores = history

# Plotting performance of model
UtilsModule.plot_losses(losses_g, losses_d, save_path='results/losses_plot.png')
UtilsModule.plot_precision(real_scores, fake_scores, save_path='results/precision_plot.png')

# Record the end time
end_time = time.time()
UtilsModule.return_time(end_time - start_time)
