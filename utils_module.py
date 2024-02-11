# utils_module.py
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

class UtilsModule:

    @staticmethod
    def to_device(data, device):
        if isinstance(data,(list,tuple)):
            return [UtilsModule.to_device(x,device) for x in data]
        return data.to(device, non_blocking=True)
    
    @staticmethod
    def save_samples(index, generator, latent_tensors, sample_dir, show=True):
        fake_images = generator(latent_tensors)
        fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
        save_image(UtilsModule.denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
        print('Saving', fake_fname)
        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

    @staticmethod
    def plot_losses(loss_g, loss_d, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(loss_g, label='Generator Loss', color='green')
        plt.plot(loss_d, label='Discriminator Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Generator and Discriminator Losses')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    @staticmethod
    def denorm(img_tensors):
        stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # Normalization
        return img_tensors * stats[1][0] + stats[0][0]

    @staticmethod
    def plot_precision(real_scores, fake_scores, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(real_scores, label='Real Average Precision', color='blue')
        plt.plot(fake_scores, label='Fake Average Precision', color='purple')
        plt.xlabel('Epochs')
        plt.ylabel('Average Precision')
        plt.title('Real and Fake Average Precision Scores')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


    @staticmethod
    def return_time(elapsed_time_seconds):
        # Convert seconds to hours, minutes, and seconds
        hours, remainder = divmod(elapsed_time_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nElapsed Time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")