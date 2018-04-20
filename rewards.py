import torch
from torch.autograd import Variable

def compute_reward(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
    """
    Compute diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """
    if isinstance(seq, Variable): seq = seq.data
    if isinstance(actions, Variable): actions = actions.data
    pick_idxs = actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs)
    if num_picks == 0:
        # give zero reward is no frames are selected
        return 0.
    seq = seq.squeeze()
    n = seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_div = 0.
    else:
        normed_seq = seq / seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t()) # dissimilarity matrix [Eq.4]
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1)) # diversity reward [Eq.3]

    # compute representativeness reward
    dist_mat = torch.pow(seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, seq, seq.t())
    dist_mat = dist_mat[:,pick_idxs]
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]

    # combine the two rewards
    reward = (reward_div + reward_rep) * 0.5

    return reward