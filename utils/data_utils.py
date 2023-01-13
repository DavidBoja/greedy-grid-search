

from torch.utils.data import DataLoader, Dataset
import torch

from utils.pc_utils import voxelize_batch


class RotatePC(Dataset):
    """
    Rotates a given centered point cloud by a given batch of rotations. 
    __getitem__ first rotates the point cloud and then makes it positive by
    translating the minimal bounding box point to the origin.
    """
    
    def __init__(self, pts, R_batch, subsampling_indices=None, center=torch.zeros(3),
                 voxelization_option=None,voxel_size=0.06):

        self.pts = pts
        self.R_batch = R_batch
        self.K = self.R_batch.shape[0]

        self.center = torch.mean(self.pts,axis=0) - center
        self.points_preprocessed = self.pts - self.center
        if not isinstance(subsampling_indices, type(None)):
            self.points_preprocessed = self.points_preprocessed[subsampling_indices]

    def __len__(self):
        return self.K

    def __getitem__(self, idx):
       
        # rotate
        points = torch.matmul(self.R_batch[idx], #  3 x 3
                              self.points_preprocessed.T).T # 3 X N

        # make positive by translating min bounding box point to origin
        minima = torch.min(points,dim=0)[0]
        points = points - minima

        # returns rotated + centered + "positived"
        return points, minima


class RotatePCcollator(object):
    """
    Collator that for a batch of points with same size, voxelizes and pads 
    them together -- __call__ returns torch tensor of  1 x N x Vx x Vy x Vz dims
    """
    
    def __init__(self, voxel_size, pp, fill_positive=5, fill_negative=-1,
                 fill_padding=-1):
        
        self.voxel_size = voxel_size
        self.pp = pp
        self.fill_positive = fill_positive
        self.fill_negative = fill_negative
        self.fill_padding = fill_padding
        
    def __call__(self,batch):
       
        # NOTE -- batch must have pc-s of same nr points
        # we use the same point cloud rotated in multiple ways so we good

        minimas = torch.stack([b[1] for b in batch],dim=0)
        pts = torch.stack([b[0] for b in batch],dim=0)

        voxelized_batch, _ = voxelize_batch(pts, 
                                        self.voxel_size,
                                        self.fill_positive,
                                        self.fill_negative) # N x Vx x Vy x Vz

        voxelized_batch_padded = torch.nn.functional.pad(voxelized_batch.type(torch.int32), 
                                                        self.pp, mode='constant', 
                                                        value=self.fill_padding)

        return voxelized_batch_padded.unsqueeze(1), minimas



def preprocess_pcj(pcj, R_batch, voxel_size, pp, batch_size, num_workers):
    """
    Create a dataloader that loads batch_size batches of rotated, voxelized  and padded pcj 
    points for a given dataset. 
    
    The batches are voxelized and paded pcj rotated with a number of
    rotations from R_batch.
  
    Input:  pcj: (torch) Nx3 points 
            R_batch: (torch) NX3x3 rotations
            voxel_size: (float) size of voxel side
            pp: (tuple) 6dim tuple for padding -- deterimend in padding.padding_options
            batch_size: (scalar) torch dataloader size of batch
            num_workers: (scalar) torch dataloader num workers
    Returns: my_data: (torch.Dataset) dataset that returns rotated pcj for given index of R_batch
             my_dataloader: (torch.DataLoader) datalodaer that loads batch_size batches of rotated, 
                                                voxelized  and padded pcj points for a given dataset. 
    """
    
    my_data = RotatePC(pcj, R_batch)
    my_data_collator = RotatePCcollator(voxel_size, pp)
    my_dataloader = DataLoader(my_data, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                collate_fn=my_data_collator,
                                num_workers=num_workers
                                    )
    

    return my_data, my_dataloader