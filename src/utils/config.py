import torch

DTYPE = torch.float32
USE_GPU = True

ACCURACY = 'accuracy'
DICE_SCORE = 'dice_score'
JACCARD_INDEX = 'jaccard_index'
PRECISION = 'precision'
RECALL = 'recall'
SPECIFICITY = 'specificity'
F1_SCORE = 'f1_score'
AUROC_ = 'auroc'
AUPRC = 'auprc'

def get_device() -> torch.device:
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    return device
