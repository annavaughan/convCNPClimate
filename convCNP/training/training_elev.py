"""
Training functions: models with MLP elevation
"""

import torch
import numpy as np 
import scipy
from .utils import log_exp, generate_context_mask, get_fold_data

def train_batch_elev(task, opt, model, ll, elev, dists):
    """
    Train one batch
    Parameters:
    -----------
    task: dict
        ['y_context', 'y_target']
    opt: Optimizer
    model: convCNP model
    ll: loss function
    """
    batch_size, channels, x, y = task['y_context'].shape

    # Generate mask
    mask = generate_context_mask(batch_size, channels, x, y)

    # Forward pass  
    v = model(task['y_context'], mask, dists, elev)
    
    # Backprop
    obj = -ll(task['y_target'], v)
    obj.backward()
    opt.step()
    opt.zero_grad()

    return obj, opt, model

def eval_epoch_elev(model, held_out, ll, elev, dists, y_target_t, get_value):
    """
    Calculate nll on held out dataset after each epoch
    """
    model.eval()

    targets = [i['y_target'] for i in held_out]
    targets_complete = torch.cat(targets, axis = 0)

    predictions = []
    with torch.no_grad():
        for task in held_out:
            batch_size, channels, x, y = task['y_context'].shape
            
            # Predict parameters for the batch
            mask = generate_context_mask(batch_size, channels, x, y)
            predictions.append(model(task['y_context'], mask, dists, elev))

    # Calculate NLL
    predictions = torch.cat(predictions)
    eval_ll = -ll(targets_complete, predictions)

    # Transform predicted parameters to amounts
    predictions = get_value(predictions)

    maes = np.zeros(predictions.shape[1])
    spearmans = np.zeros(predictions.shape[1])
    pearsons = np.zeros(predictions.shape[1])

    # Print output by station
    for st in range(predictions.shape[1]):
        true_mean = targets_complete[:, st].detach().cpu().numpy() #y_target_t.inverse_transform(targets_complete[:, st].view(-1, 1).cpu())
        pred_mean = predictions[:, st].detach().cpu().numpy() #y_target_t.inverse_transform(predictions[:, st].view(-1, 1).detach().cpu())
        pred_mean = pred_mean[~np.isnan(true_mean)]
        true_mean = true_mean[~np.isnan(true_mean)]
        try:
            maes[st] = np.mean(np.abs(true_mean - pred_mean))
            pearsons[st] = scipy.stats.pearsonr(pred_mean, true_mean)[0]
            spearmans[st] = scipy.stats.spearmanr(pred_mean, true_mean).correlation
        except:
            maes[st] = np.nan
            pearsons[st] = np.nan
            spearmans[st] = np.nan
            continue
        #plt.plot(true_mean)
        #plt.plot(pred_mean)
        #plt.show()


    print("Mean absolute error: {}".format(np.median(maes[~np.isnan(maes)])))
    print("Pearson correlation: {}".format(np.median(pearsons[~np.isnan(pearsons)])))
    print("Spearman correlation: {}".format(np.median(spearmans[~np.isnan(spearmans)])))
    
    return eval_ll

def train_epoch_elev(model, opt, training_data, ll, elev, dists):
    """
    Outer training loop for each epoch
    """
    model.train()

    # Train and update the model
    batch_objs = []
    for task in training_data:
        # Generate a mask
        obj, opt, model = train_batch_elev(task, opt, model, ll, elev, dists)
        batch_objs.append(np.float(obj.item()))
    train_ll = np.mean(np.array(batch_objs)[-5:])
    
    return train_ll
            
def train_elev(model, 
          opt, 
          ll,
          elev,
          dists,
          y_context,
          y_target,
          output_dir,
          y_target_t,
          get_value, 
          fold,
          n_epochs = 100):
    """
    Top level training loop for the model
    """
    test_score = []

    best_obj = 5

    # Run the training loop.
    print("Training")

    for epoch in range(n_epochs):
        print("Starting epoch: {}".format(epoch))
        if epoch>0:
            del training_data
            del held_out
        training_data, held_out = get_fold_data((8766, 10958), y_context, y_target)

        # Compute training objective.
        train_obj = train_epoch_elev(model, opt, training_data, ll, elev, dists)
        test_obj = eval_epoch_elev(model, held_out, ll, elev, dists, y_target_t, get_value)
        test_score.append(test_obj)
        print('Epoch %s: train NLL %.3f, test NLL %.3f' % (epoch, train_obj, test_obj))

        if test_obj < best_obj:
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': test_score}, output_dir+"model_fold_{}".format(fold))
            best_obj = test_obj