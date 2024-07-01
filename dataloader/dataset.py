from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, source_arr, target_arr):
        super().__init__()

        num_sample = len(source_arr)

        self.source_arr = source_arr
        self.target_arr = target_arr
        self.num_sample = num_sample

    def __getitem(self, idx):
        return self.source_arr[idx], self.source_arr[idx]

    def __len__(self):
        return self.num_sample