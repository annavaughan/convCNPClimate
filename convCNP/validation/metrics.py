"""
Metrics for comparison to VALUE data
"""

import numpy as np 
from scipy.stats import ks_2samp, spearmanr, pearsonr
from statsmodels.tsa.stattools import acf


def r01(real, predicted):
    """
    Compare quantile exceedence for real and predicted 
    data for quantile p
    """
    n_wet_real = np.where(real>=0.2)[0].shape[0]
    n_wet_predicted = np.where(predicted>=0.2)[0].shape[0]

    return n_wet_predicted/n_wet_real

def sd2(real, predicted):
    """
    Test whether two samples are drawn from the same distribution
    returns ks p-value
    """
    real_mean = np.mean(real[real>=0.2])
    pred_mean = np.mean(predicted[predicted>=0.2])
    return pred_mean - real_mean

def correlations(real, predicted):
    """
    Returns the Spearman and Pearson correlations for observed and predicted
    """
    sp = spearmanr(predicted, real).correlation
    pr = pearsonr(predicted, real)[0]
    return sp, pr

def mae(real, predicted):
    """
    Mean absolute error
    """
    return np.mean(np.abs(real-predicted))

def mean_bias(real, predicted):
    return np.mean(real) - np.mean(predicted)

def ks_test(real, predicted):
    """
    Test whether two samples are drawn from the same distribution
    returns ks p-value
    """
    return ks_2samp(real, predicted)[0]

def ww_probability(real, predicted):
    """
    Probability of wet-wet transition
    Wet defined as >1mm
    """
    real_inds = np.zeros(real.shape)
    pred_inds = np.zeros(predicted.shape)

    real_inds[real>1] = 1
    pred_inds[predicted>1] = 1

    real_ww_count = 0
    pred_ww_count = 0

    for i in range(predicted.shape[0]-1):
        if np.logical_and(pred_inds[i]==1, pred_inds[i+1]==1):
            pred_ww_count+=1
        if np.logical_and(real_inds[i]==1, real_inds[i+1]==1):
            real_ww_count+=1

    real_ww_prob = real_ww_count/(real.shape[0]-1)
    pred_ww_prob = pred_ww_count/(predicted.shape[0]-1)

    return pred_ww_prob/real_ww_prob

def wd_probability(real, predicted):
    """
    Probability of wet-dry transition
    wet defined as >1mm
    """
    real_inds = np.zeros(real.shape)
    pred_inds = np.zeros(predicted.shape)

    real_inds[real>1] = 1
    pred_inds[predicted>1] = 1

    real_wd_count = 0
    pred_wd_count = 0

    for i in range(predicted.shape[0]-1):
        if np.logical_and(pred_inds[i]==1, pred_inds[i+1]==0):
            pred_wd_count+=1
        if np.logical_and(real_inds[i]==1, real_inds[i+1]==0):
            real_wd_count+=1

    real_wd_prob = real_wd_count/(real.shape[0]-1)
    pred_wd_prob = pred_wd_count/(predicted.shape[0]-1)

    return pred_wd_prob/real_wd_prob

def get_spell_length(arr, val):
    """
    Get lengths of dry (val=0) or wet (val=1) spells
    """
    ind = 0
    real_wet_spell_length = []
    while ind<arr.shape[0]:
        current_val = arr[ind]
        if current_val == val:
            days = 0
            spell_count = ind
            while arr[spell_count] == val:
                spell_count+=1
                days+=1
                if spell_count>=arr.shape[0]:
                    break
            real_wet_spell_length.append(days)
            ind+=days
            continue
        ind+=1
    return np.array(real_wet_spell_length)

def wet_spell_mb(real, predicted):
    """
    Calculate bias in 50th percentile of wet spell duration
    """

    real_inds = np.zeros(real.shape)
    pred_inds = np.zeros(predicted.shape)

    real_inds[real>1] = 1
    pred_inds[predicted>1] = 1

    real_wet_spell_length = get_spell_length(real_inds, 1)
    pred_wet_spell_length = get_spell_length(pred_inds, 1)

    p50_real = np.mean(np.array(real_wet_spell_length))
    p50_pred = np.mean(np.array(pred_wet_spell_length))

    return p50_pred - p50_real

def dry_spell_mb(real, predicted):
    """
    Calculate bias in 50th percentile of dry spell duration
    """

    real_inds = np.zeros(real.shape)
    pred_inds = np.zeros(predicted.shape)

    real_inds[real>1] = 1
    pred_inds[predicted>1] = 1

    real_dry_spell_length = get_spell_length(real_inds, 0)
    pred_dry_spell_length = get_spell_length(pred_inds, 0)

    p50_real = np.mean(np.array(real_dry_spell_length))
    p50_pred = np.mean(np.array(pred_dry_spell_length))

    return p50_pred - p50_real

def r10(real, predicted):
    """
    Calculate R10
    """
    real_r10 = real[real>10].shape[0]
    pred_r10 = predicted[predicted>10].shape[0]
    return pred_r10/real.shape[0] - real_r10/real.shape[0]

def P98Wet(real, predicted):
    """
    Calculate the 98th percentile of precipitation
    """
    real_p98 = np.quantile(real[real>1], 0.98)
    pred_p98 = np.quantile(predicted[predicted>1], 0.98)
    return pred_p98 - real_p98

def P98Wet_amount(real, predicted):
    """
    Calculate ratio of predicted:observed precipitation for days 
    above the 98th percentile
    """
    real_p98_amount = np.sum(real[real>np.quantile(real[real>1], 0.98)])
    pred_p98_amount = np.sum(predicted[predicted>np.quantile(predicted[predicted>1], 0.98)])
    return pred_p98_amount/real_p98_amount

def warm_spell_mean(real, predicted, v):
    """
    bias in 50th percentile of dry spell duration
    """

    real_inds = np.zeros(real.shape)
    pred_inds = np.zeros(predicted.shape)

    real_inds[real>np.quantile(real, 0.9)] = 1
    pred_inds[predicted>np.quantile(real, 0.9)] = 1

    real_warm_spell_length = get_spell_length(real_inds, 1)
    pred_warm_spell_length = get_spell_length(pred_inds, 1)

    p50_real = np.quantile(real_warm_spell_length, v)
    p50_pred = np.quantile(pred_warm_spell_length, v)

    return p50_pred - p50_real

def lag_one_autocorr(real, predicted):
    """
    Calculate bias in lag one autocorrelation
    """
    real_ac = acf(real, nlags=2)
    pred_ac = acf(predicted, nlags=2)
    return real_ac[1] - pred_ac[1]

def lag_two_autocorr(real, predicted):
    """
    Calculate bias in lag one autocorrelation
    """
    real_ac = acf(real, nlags=3)
    pred_ac = acf(predicted, nlags=3)
    return real_ac[2] - pred_ac[2]