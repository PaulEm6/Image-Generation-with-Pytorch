import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torch.utils.data import DataLoader

class DataLoaderModule:
    @staticmethod
    def create_datastorage(path, image_size, stats):
        return ImageFolder(path, transform=tt.Compose([
            tt.Resize(image_size),
            tt.CenterCrop(image_size),
            tt.ToTensor(),
            tt.Normalize(*stats),
            tt.RandomHorizontalFlip(p=0.5)
        ]))

    @staticmethod
    def create_dataloader(dataset, batch_size, shuffle=True):
        return DataLoader(dataset, batch_size, shuffle=shuffle)
    
    @staticmethod
    def get_default_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
        
    @staticmethod
    def to_device(data, device):
        if isinstance(data, (list, tuple)):
            return [DataLoaderModule.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        self.data_module = DataLoaderModule()

    def __iter__(self):
        for b in self.dl:
            yield self.data_module.to_device(data=b, device=self.device)

    def __len__(self):
        return len(self.dl)