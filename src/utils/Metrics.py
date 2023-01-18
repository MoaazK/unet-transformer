import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision, Dice, F1Score, JaccardIndex, Precision, Recall, Specificity

from utils.config import ACCURACY, AUPRC, AUROC_, DICE_SCORE, F1_SCORE, JACCARD_INDEX, PRECISION, RECALL, SPECIFICITY

class Metrics():
    def __init__(self, device: torch.device, num_classes=2) -> None:
        self.accuracy = Accuracy().to(device=device)
        self.dice_score = Dice().to(device=device)
        self.jaccard_index = JaccardIndex(num_classes=num_classes).to(device=device)
        self.precision = Precision().to(device=device)
        self.recall = Recall().to(device=device)
        self.specificity = Specificity().to(device=device)
        self.f1_score = F1Score().to(device=device)
        # self.auroc = AUROC().to(device=device)
        # self.auprc = AveragePrecision().to(device=device)

        self.num_classes = num_classes
        self.metrics_history = []

    def reset(self) -> None:
        self.accuracy.reset()
        self.dice_score.reset()
        self.jaccard_index.reset()
        self.precision.reset()
        self.recall.reset()
        self.specificity.reset()
        self.f1_score.reset()
        # self.auroc.reset()
        # self.auprc.reset()

    def update(self, input: torch.Tensor, target: torch.Tensor) -> dict:
        metric_dict = {}
        target = target.int()
        
        metric_dict[ACCURACY] = self.accuracy(input, target)
        metric_dict[DICE_SCORE] = self.dice_score(input, target)
        metric_dict[JACCARD_INDEX] = self.jaccard_index(input, target)
        metric_dict[PRECISION] = self.precision(input, target)
        metric_dict[RECALL] = self.recall(input, target)
        metric_dict[SPECIFICITY] = self.specificity(input, target)
        metric_dict[F1_SCORE] = self.f1_score(input, target)
        # metric_dict[AUROC_] = self.auroc(input, target)
        # metric_dict[AUPRC] = self.auprc(input, target)

        return metric_dict

    def compute(self) -> dict:
        metric_dict = {}

        metric_dict[ACCURACY] = self.accuracy.compute()
        metric_dict[DICE_SCORE] = self.dice_score.compute()
        metric_dict[JACCARD_INDEX] = self.jaccard_index.compute()
        metric_dict[PRECISION] = self.precision.compute()
        metric_dict[RECALL] = self.recall.compute()
        metric_dict[SPECIFICITY] = self.specificity.compute()
        metric_dict[F1_SCORE] = self.f1_score.compute()
        # metric_dict[AUROC_] = self.auroc.compute()
        # metric_dict[AUPRC] = self.auprc.compute()

        self.metrics_history.append(metric_dict)
        return metric_dict

    def get_metrics(self) -> list:
        return self.metrics_history
