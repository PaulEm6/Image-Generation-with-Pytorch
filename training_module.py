# training_module.py
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from utils_module import UtilsModule


class TrainingModule:

    @staticmethod
    def train_discriminator(generator, discriminator, device, real_images, opt_d, batch_size, latent_size):
        # Clear discriminator gradients
        opt_d.zero_grad()

        # Pass real images through discriminator
        real_preds = discriminator(real_images)
        real_targets = torch.ones_like(real_preds, device=device)
        real_loss = F.binary_cross_entropy(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()

        # Generate fake images
        latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
        fake_images = generator(latent)

        # Pass fake images through discriminator
        fake_preds = discriminator(fake_images)
        fake_targets = torch.zeros_like(fake_preds, device=device)
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        # Update discriminator weights
        loss = real_loss + fake_loss
        loss.backward()
        opt_d.step()
        return loss.item(), real_score, fake_score
    
    @staticmethod
    def train_generator(generator, discriminator, opt, device, batch_size, latent_size):
        opt.zero_grad()
        latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
        fake_images = generator(latent)
        preds = discriminator(fake_images)
        targets = torch.ones_like(preds, device=device)
        loss = F.binary_cross_entropy(preds, targets)
        loss.backward()
        opt.step()
        return loss.item()
    
    @staticmethod
    def fit(generator, discriminator, train_dl, fixed_latent, device, epochs, lr, batch_size, latent_size, start_idx=1):
        torch.cuda.empty_cache()

        # Losses & scores
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []

        # Create optimizers
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

        for epoch in range(epochs):
            for real_images, _ in tqdm(train_dl):
                # Train discriminator
                loss_d, real_score, fake_score = TrainingModule.train_discriminator(generator, discriminator, device, real_images, opt_d, batch_size, latent_size)
                # Train generator
                loss_g = TrainingModule.train_generator(generator, discriminator, opt_g, device, batch_size, latent_size)

            # Record losses & scores
            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)

            # Log losses & scores (last batch)
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch + 1, epochs, loss_g, loss_d, real_score, fake_score))

            # Save generated images
            UtilsModule.save_samples(index=epoch + start_idx, generator=generator, latent_tensors= fixed_latent, sample_dir = 'generated', show=False)

        return losses_g, losses_d, real_scores, fake_scores