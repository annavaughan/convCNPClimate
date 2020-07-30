import torch
import numpy as np
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal

def shuffle_data(context, task):
    """
    Shuffle data before each epoch
    """
    inds = np.linspace(0, context.shape[0]-1, context.shape[0])
    np.random.shuffle(inds)
    return context[inds], task[inds]

def get_fold_data(inds, context, targets, shuffle = True):
    """
    Split the context and target data into folds
    """
    # Get held out
    held_out_context = context[inds[0]:inds[1], ...]
    held_out_targets = targets[inds[0]:inds[1], :]

    # Get training
    train_context = torch.cat(
        [context[0:inds[0],...], context[inds[1]:,...]]
    )
    train_targets = torch.cat(
        [targets[0:inds[0],...], targets[inds[1]:,...]]
    )

    # Shuffle data
    if shuffle: 
        train_context, train_targets = shuffle_data(train_context, train_targets)
        held_out_context, held_out_targets = shuffle_data(held_out_context, held_out_targets)
    

    train_targets = torch.split(train_targets, 16)
    train_context = torch.split(train_context, 16)
    held_out_context = torch.split(held_out_context, 16)
    held_out_targets = torch.split(held_out_targets, 16)

    training_data = [{"y_context":train_context[i], "y_target":train_targets[i]} 
        for i in range(len(train_targets))]
    held_out = [{"y_context":held_out_context[i], "y_target":held_out_targets[i]} 
        for i in range(len(held_out_targets))]

    return training_data, held_out

def make_r_mask(target_vals):
    """
    Make the r mask for the Bernoulli precipitation distribution
    """
    # Make r mask
    r = torch.ones(target_vals.shape[0]).cuda()
    r[target_vals==0] = 0
    
    # Set the target vals to one to stop the pesky error
    # (It doesn't contribute anyway)
    target_vals[target_vals == 0] = 0.01

    return r, target_vals

def log_exp(x):
    """
    Fix overflow
    """
    lt = torch.where(torch.exp(x)<1000)
    if lt[0].shape[0] > 0:
        x[lt] = torch.log(1+torch.exp(x[lt]))
    return x

def generate_context_mask(batch_size, n_channels, x, y):
    """
    Generate a context mask - in this simple case this will be one 
    for all grid points
    """
    return torch.ones(batch_size, n_channels, x, y).cuda()

def get_value_pr_gammagp(p):
    """
    Return predicted mean to calculate stats each epoch
    """

    # output.shape = [time, stations]
    output = torch.zeros(p.shape[0], 3010).cuda()

    for st in range(3010):
        # For each need to calculate the normalisation 
        x = torch.linspace(0.2, 150, 1499)
        x = x.view(-1, 1).repeat(1, p.shape[0]).cuda()
        norms = torch.sum(tbi_func(x, p[:, st, :]), dim = 0)
        ev = torch.sum(x*(1/norms)*tbi_func(x, p[:, st, :]), dim = 0)       
        output[:, st][p[:,st,0]>0.5] = ev

    return output

def get_value_tmax(p):
    """
    Return predicted mean to calculate stats each epoch
    """
    return p[:, :, 0]

def tbi_func(x, v):
    """
    Evaluate Bernoulli-gamma-GP mixture likelihood
    Parameters:
    ----------
    v: torch.Tensor(batch*86, channels)
        parameters from model
    x: torch.Tensor(batch*86)
        target vals to eval at
    """
    # Gamma distribution
    g = Gamma(concentration = v[:,2], rate = v[:,3])
    gamma = torch.exp(torch.clamp(g.log_prob(x), min=-1e5, max=1e5))

    # Weight term
    weight_term = (1/2)+(1/np.pi)*torch.atan((x-v[:,5])/v[:,6])

    # GP distribution
    gp = (1/v[:,4])*(1+(v[:,1]*x/v[:,4]))**((-1/v[:,1])-1)

    # total
    tbi = gamma*(1-weight_term)+gp*weight_term
    return torch.clamp(tbi, min = 1e-5)
