
import numpy as np


def RRE(R_gt,R_estim):
    '''
    Calc. relative rotation error for 2 rotation matrices.

    Parameters:
    :param R_gt: numpy array dim (3,3)
    R_estim: np array dim (3,3)
    
    Returns: angle measurement in degrees
    '''

    # tnp = np.matmul(R_estim.T,R_gt)
    tnp = np.matmul(np.linalg.inv(R_estim),R_gt)
    tnp = (np.trace(tnp) -1) /2
    tnp = np.clip(tnp, -1, 1)
    tnp = np.arccos(tnp) * (180/np.pi)
    return tnp

def RTE(t_gt,t_estim):
    '''
    t_gt: np array dim (3,)
    t_estim: np array dim (3,)
    '''

    return np.linalg.norm(t_gt - t_estim,ord=2)


def eval_from_csv(data,thr_rot=15,thr_trans=0.3,M=1724,rre_col='RRE'):

    true_positives = (data['RTE'] < thr_trans) & (data[rre_col] < thr_rot)
    print(f'Registered {np.sum(true_positives)}/{data.shape[0]} examples.')
    print(f'RR: {np.sum(true_positives)/data.shape[0]:.4f}')

    rre_mean = np.mean(data.loc[true_positives,rre_col])
    print(f'RRE: {rre_mean:.4f} degrees')

    rte_mean = np.mean(data.loc[true_positives,'RTE']) * 100
    print(f'RTE: {rte_mean:.4f} cm')


    N = data.shape[0]
    print(f'Results obtained on {(N/M)*100}% of benchmark examples')