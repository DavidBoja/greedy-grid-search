
import argparse
import os.path as osp
import yaml
import time
import torch
import pandas as pd
from datetime import timedelta
import numpy as np
from pprint import pprint
import open3d as o3d
import json
import os
import pickle

from fft_conv_pytorch import fft_conv

from utils.pc_utils import voxelize, unravel_index_pytorch
from utils.data_utils import preprocess_pcj
from utils.rot_utils import create_T_estim_matrix, load_rotations
from utils.utils import set_seeds
from utils.padding import padding_options

def load_point_cloud(pc_path):

    # check if known type of pc
    pc_extension = pc_path.split('.')[-1]
    choices = ['ply','csv']
    assert pc_extension in choices, f'Can only load {choices} files, cant load {pc_extension}'

    if pc_extension == 'csv':
        pc = pd.read_csv(pc_path,header=None)
        pc = np.array(pc)
        pc = torch.from_numpy(pc)
    elif pc_extension == 'ply':
        pc = o3d.io.read_point_cloud(pc_path) 
        pc = np.asarray(pc.points)
        pc = torch.from_numpy(pc)

    return pc


def register(config):
    '''
    Register selected dataset.
    '''
    
    pprint(config)
    DEVICE = torch.device('cuda:{}'.format(config['GPU-INDEX']))
    PADDING = config['PADDING']
    BATCH_SIZE = config['BATCH-SIZE']
    VOXEL_SIZE = config['VOXEL-SIZE']
    PV = config['PV']
    NV = config['NV']
    PPV = config['PPV']
    NUM_WORKERS = config['NUM-WORKERS']
    ROTATION_CHOICE = config['ROTATION-OPTION']
    PCI_PATH = config['PCI-PATH'] 
    PCJ_PATH = config['PCJ-PATH'] 

    # create results paths and files
    now = time.strftime('%y%m%d%H%M')
    results_folder_path = f'results/{now}'
    if not os.path.exists('results'):
        os.mkdir('results')
    os.mkdir(f'results/{now}')
   
    # load inputs
    pci = load_point_cloud(PCI_PATH)
    pcj = load_point_cloud(PCJ_PATH)
    R_batch = load_rotations(ROTATION_CHOICE)

    init_time = time.time()
    
    #### PREPROCESS pci ##########################################################
    print('Preprocessing...')
    # 1. make pci positive for voxelization
    make_pci_posit_translation = torch.min(pci,axis=0)[0]
    pci = pci - make_pci_posit_translation
    
    # 2. voxelize pci
    pci_voxel, NR_VOXELS_PCI = voxelize(pci, VOXEL_SIZE,
                                            fill_positive=PV,
                                            fill_negative=NV)

    # find indices of the pci central voxel 
    CENTRAL_VOXEL_PCI = torch.where(NR_VOXELS_PCI % 2 == 0, # check if even
                                    (NR_VOXELS_PCI / 2) -1, # if even take one voxel to the left 
                                    torch.floor(NR_VOXELS_PCI / 2)).int() # else just take middle voxel
    # find central voxel in xyz coordinates
    central_voxel_center =  CENTRAL_VOXEL_PCI * VOXEL_SIZE + (0.5*VOXEL_SIZE)

    # 3. move pci on cuda -- dims needed 1 x 1 x Vx x Vy x Vz
    weight_to_fftconv3d = pci_voxel.type(torch.int32).to(DEVICE)[None,None,:,:,:]

    #### PREPROCESS pcj = target ##########################################################
    # define padding (z,y,x) axis is the order for padding
    pp, pp_xyz = padding_options(PADDING,
                                    CENTRAL_VOXEL_PCI,
                                    NR_VOXELS_PCI)

    # batch pcj voxelized data
    my_data, my_dataloader = preprocess_pcj(pcj, 
                                            R_batch, 
                                            VOXEL_SIZE, 
                                            pp, 
                                            BATCH_SIZE,
                                            NUM_WORKERS)

    preprocess_time = time.time() - init_time

    #### PROCESS (FFT) #####################################################################
    print('Processing...')
    maxes = []
    argmaxes = []
    shapes = []
    minimas = torch.empty(R_batch.shape[0],3)

    fft_iter_time = time.time()
    for ind_dataloader,(voxelized_batch_padded,mins) in enumerate(my_dataloader):
        minimas[ind_dataloader * BATCH_SIZE:
                (ind_dataloader+1) * BATCH_SIZE,:] = mins
        
        input_to_fftconv3d = voxelized_batch_padded.to(DEVICE)
        
        out = fft_conv(input_to_fftconv3d, 
                        weight_to_fftconv3d, bias=None)
        
        maxes.append(torch.max(out))
        argmaxes.append(torch.argmax(out))
        shapes.append(out.shape)

    fft_iter_time = time.time() - fft_iter_time

    #### POST-PROCESS ####################################################################
    print('Post-processing...')
    post_process_time = time.time()
    # 1. find voxel location with biggest cross-correlation value
    m_index = torch.argmax(torch.stack(maxes)) # tells us which batch had max response
    ind0, _, ind1, ind2, ind3 = unravel_index_pytorch(argmaxes[m_index], 
                                                        shapes[m_index])

    # when batch_size = 1, this equals to m_index
    rotation_index = m_index * BATCH_SIZE + ind0
    R = R_batch[rotation_index]

    # translation -- translate for padding pp_xyz and CENTRAL_VOXEL_PCI
    # and then in the found max cc voxel
    t = torch.Tensor([-(pp_xyz[0] * VOXEL_SIZE) + 
                        ((CENTRAL_VOXEL_PCI[0]) * VOXEL_SIZE) +
                        (ind1 * VOXEL_SIZE) + 
                        (0.5 * VOXEL_SIZE),
                        
                        -(pp_xyz[2] * VOXEL_SIZE) + 
                        ((CENTRAL_VOXEL_PCI[1]) * VOXEL_SIZE) +
                        (ind2 * VOXEL_SIZE) + 
                        (0.5 * VOXEL_SIZE),
                        
                        -(pp_xyz[4] * VOXEL_SIZE) + 
                        ((CENTRAL_VOXEL_PCI[2]) * VOXEL_SIZE) +
                        (ind3 * VOXEL_SIZE) + 
                        (0.5 * VOXEL_SIZE)
                        ])

    center_pcj_translation = my_data.center
    make_pcj_posit_translation = minimas[rotation_index]
    estim_T_baseline = create_T_estim_matrix(center_pcj_translation,
                                                R,
                                                make_pcj_posit_translation,
                                                central_voxel_center,
                                                t,
                                                make_pci_posit_translation
                                                )
                                                
    post_process_time = time.time() - post_process_time
    baseline_times = time.time() - init_time


    #### SAVE SOLUTION  #############################################################
    print(f'Saving results in results/{now} ...')
    config['estimated-transformation'] = estim_T_baseline
    config['preprocess-time'] = timedelta(seconds=preprocess_time).__str__()
    config['process-time'] = timedelta(seconds=fft_iter_time).__str__()
    config['postprocess-time'] = timedelta(seconds=post_process_time).__str__()
    config['baseline-time'] = timedelta(seconds=baseline_times).__str__()

    save_results_pth = osp.join(results_folder_path,'results_and_parameters.pickle')
    with open(save_results_pth,'wb') as f:
        pickle.dump(config,f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done!')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pci_pth", 
                        required=True,
                        type=str, 
                        help='Path to target point cloud ply/csv file')
    parser.add_argument("--pcj_pth", 
                        required=True,
                        type=str, 
                        help='Path to source point cloud ply/csv file')
    args = parser.parse_args()

    # set options
    with open('config.yaml','r') as f:
        config = yaml.safe_load(f)
    config = config['DEMO']

    config['PCI-PATH'] = args.pci_pth
    config['PCJ-PATH'] = args.pcj_pth
    
    if config['SET-SEED']:
        set_seeds()

    register(config)