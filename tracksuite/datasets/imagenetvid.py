from torch.utils.data.dataset import Dataset

class ImageNetVIDDataSet(Dataset):
    def __init__(self, path, mode='train'):
        self.path = path
        