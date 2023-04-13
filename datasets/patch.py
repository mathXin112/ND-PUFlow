import torch
from torch.utils.data import Dataset
import pytorch3d.ops
from tqdm.auto import tqdm


def make_patches_for_pcl_pair(pcl_low, pcl_high, patch_size, num_patches, ratio):
    """
    Args:
        pcl_low:  Low-resolution point cloud, (N, 3).
        pcl_high: High-resolution point cloud, (rN, 3).
        patch_size:   Patch size M.
        num_patches:  Number of patches P.
        ratio:    Ratio r.
    Returns:
        (P, M, 3), (P, rM, 3)
    """
    N = pcl_low.size(0)
    seed_idx = torch.randperm(N)[:num_patches]   # (P, )
    seed_pnts = pcl_low[seed_idx].unsqueeze(0)   # (1, P, 3)
    _, _, pat_low = pytorch3d.ops.knn_points(seed_pnts, pcl_low.unsqueeze(0), K=patch_size, return_nn=True)
    pat_low = pat_low[0]    # (P, M, 3)
    _, _, pat_high = pytorch3d.ops.knn_points(seed_pnts, pcl_high.unsqueeze(0), K=ratio*patch_size, return_nn=True)
    pat_high = pat_high[0]
    return pat_low, pat_high
    

class PairedPatchDataset(Dataset):

    def __init__(self, datasets, ratio, patch_size=256, num_patches=200, transform=None):
        super().__init__()
        self.datasets = datasets
        self.ratio = ratio
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.transform = transform
        self.patches = []
        # Initialize
        self.make_patches()

    def make_patches(self):
        for dataset in tqdm(self.datasets, desc='MakePatch'):
            # pat_low, pat_high = make_patches_for_pcl_pair(
            #     data['pcl_low'],
            #     data['pcl_high'],
            #     patch_size=self.patch_size,
            #     num_patches=self.num_patches,
            #     ratio=self.ratio
            # )  # (P, M, 3), (P, rM, 3)
            # for i in range(pat_low.size(0)):
            #     self.patches.append((pat_low[i], pat_high[i],))
            # print("dataset", dataset)
            for data in tqdm(dataset):
                # print("data", data)
                pat_low, pat_high = make_patches_for_pcl_pair(
                    data['pcl_low'],
                    data['pcl_high'],
                    patch_size=self.patch_size,
                    num_patches=self.num_patches,
                    ratio=self.ratio
                )   # (P, M, 3), (P, rM, 3)
                for i in range(pat_low.size(0)):
                    self.patches.append((pat_low[i], pat_high[i], ))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        data = {
            'pcl_low': self.patches[idx][0].clone(), 
            'pcl_high': self.patches[idx][1].clone(),
        }
        if self.transform is not None:
            data = self.transform(data)
        return data
