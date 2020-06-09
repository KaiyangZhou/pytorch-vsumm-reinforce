import torch

def compute_reward(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
    """
    Compute Diversity reward and Representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU

    """

    _seq = seq.detach()
    _actions = actions.detach()

    # get selected frames indices
    pick_indices = actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_indices) if pick_indices.ndimension() > 0 else 1

    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0, )
        if use_gpu: reward = reward.cuda()
        return reward

    _seq = seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    # Rdiv = 1 / (Y * (Y-1)) * SUM(SUM( d(xt,xt') ))
    # d(xt,xt') = 1 - ( xtT*xt' /  (||xt|| * ||xt'||) )
    if num_picks == 1:
        reward_div = torch.tensor(0, )
        if use_gpu: reward_div.cuda()

    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1 - torch.matmul(normed_seq, normed_seq.t()) # dissimilarity matrix [Eq.4]

        # Y : Selected frames indices
        # pick_idx : Y
        dissim_submat = dissim_mat[pick_indices, :][: ,pick_indices]

        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_indices.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.

        reward_div = dissim_submat.sum() / (num_picks* (num_picks - 1.)) # diversity reward [Eq.3]

    # compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())

    dist_mat = dist_mat[:, pick_indices]
    dist_mat = dist_mat.min(1, keepdim=True)[0]

    reward_rep = torch.exp(-dist_mat.mean()) # representativeness reward [Eq.5]

    reward = (reward_div + reward_rep) * 0.5

    return reward