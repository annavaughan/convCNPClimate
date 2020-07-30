import torch

def generate_context_mask(batch_size, n_channels, x, y):
    """
    Generate a context mask - in this simple case this will be one 
    for all grid points
    """
    return torch.ones(batch_size, n_channels, x, y).cuda()

def get_dists(target_x, grid_x, grid_y):
    """
    Get the distances between the grid points and true points
    """
    # dimensions
    x_dim, y_dim = grid_x.shape

    # Init large scale grid
    total_grid = torch.zeros(target_x.shape[0], x_dim, y_dim)
    count = 0

    for point in target_x:
        # Calculate distance from point to each grid
        dists = (grid_x - point[0])**2+(grid_y - point[1])**2
        total_grid[count, :, :] = dists

        count += 1

    return total_grid.cuda()

def log_exp(x):
    """
    Fix overflow
    """
    lt = torch.where(torch.exp(x)<1000)
    if lt[0].shape[0] > 0:
        x[lt] = torch.log(1+torch.exp(x[lt]))
    return x

def force_positive(x):
    return 0.01+ (1-0.1)*log_exp(x)