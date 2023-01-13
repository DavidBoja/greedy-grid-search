
import torch
import operator


def unravel_index_pytorch(flat_index, shape): 
    flat_index = operator.index(flat_index) 
    res = [] 

    # Short-circuits on zero dim tensors 
    if shape == torch.Size([]): 
        return 0 

    for size in shape[::-1]: 
        res.append(flat_index % size) 
        flat_index = flat_index // size 

    if len(res) == 1: 
        return res[0] 

    return tuple(res[::-1])
    
def voxelize(points, voxel_size, fill_positive=1, fill_negative=0):
    """
    Voxelize points to voxel_size.
  
    Input:  points: (torch) Nx3 points to voxelize 
            voxel_size: (int) scalar that determines size of one voxel
            fill_positive: (int) number put in place of filled voxels
            fill_negative: (int) number put in place of emtpy voxels
    Returns: voxels (torch): voxelized points of dim 
                            NR_VOXELS[0] x NR_VOXELS[1] x NR_VOXELS[2]
             NR_VOXELS: (torch) tensor of voxel dimensions, dim3
    """

    # max of input by ax
    max_ax_input = torch.max(points,dim=0)[0]
    NR_VOXELS = (torch.floor(max_ax_input/voxel_size) + 1).type(torch.int64)

    voxels = torch.zeros(tuple(NR_VOXELS.tolist())) + fill_negative
    voxel_indices = torch.floor(points / voxel_size).long()
    voxels[voxel_indices[:,0],
           voxel_indices[:,1],
           voxel_indices[:,2]] = fill_positive
    
    return voxels, NR_VOXELS

def voxelize_batch(points, voxel_size, fill_positive=1, fill_negative=0):
    """
    Voxelize multiple point clouds and batch the voxelized volumes.
    Because of batching, each point cloud needs to be voxelized into 
    same volume dimensions.
  
    Input:  points: (torch) BxNx3 points to voxelize 
            voxel_size: (int) scalar that determines size of one voxel
            fill_positive: (int) number put in place of filled voxels
            fill_negative: (int) number put in place of emtpy voxels
    Returns: voxels (torch): voxelized points of dim 
                            B?? x NR_VOXELS[0] x NR_VOXELS[1] x NR_VOXELS[2]
             NR_VOXELS: (torch) tensor of voxel dimensions, dim3
    """

    # max of input by ax
    max_ax_input = torch.max(torch.max(points,dim=1)[0],dim=0)[0]
    # print_time_elapsed(tt1,'max_ax_input','green')

    # tt2 = time.time()
    NR_VOXELS = (torch.floor(max_ax_input/voxel_size) + 1).type(torch.int64).tolist()
    # print_time_elapsed(tt2,'NR_VOXELS','green')

    # tt3 = time.time()
    B = points.shape[0]
    N = points.shape[1]
    dims = tuple([B] + NR_VOXELS)
    # print_time_elapsed(tt3,'dims','green')

    # tt4 = time.time()
    voxels = torch.zeros(dims) + fill_negative
    # print_time_elapsed(tt4,'create voxels and fill negative','green')

    # tt5 = time.time()
    voxel_indices = torch.floor(points / voxel_size).type(torch.int16)
    # print_time_elapsed(tt5,'voxel_indices','green')

    # reshape tako da bude N x 4
    # tt6 = time.time()
    batch_index = torch.arange(B,dtype=torch.int16).repeat(N,1).transpose(0,1)
    # print_time_elapsed(tt6,'batch_index','green')

    # tt7 = time.time()
    # batch_voxel_indices = torch.cat([batch_index[:,:,None].int(),
    #                                  voxel_indices.int()],dim=2)
    batch_voxel_indices = torch.cat([batch_index.unsqueeze(-1),
                                     voxel_indices],dim=2)
    # print_time_elapsed(tt7,'batch_voxel_indices','green')

    # tt8 = time.time()
    batch_voxel_indices = batch_voxel_indices.reshape(-1,4).long()
    # print_time_elapsed(tt8,'batch_voxel_indices','green')

    # tt9 = time.time()
    voxels[batch_voxel_indices[:,0],
           batch_voxel_indices[:,1],
           batch_voxel_indices[:,2],
           batch_voxel_indices[:,3]] = fill_positive
    # print_time_elapsed(tt9,'voxels fill positive','green')

    return voxels, NR_VOXELS