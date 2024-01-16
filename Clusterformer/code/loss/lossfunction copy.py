import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

class CrossEntropy():
    def __init__(self, logits, labels):
        super(CrossEntropy, self).__init__()
        self.logits = logits
        self.labels =  labels
    def loss_function(self):
        return F.cross_entropy(self.logits,self.labels)
    

class FocalLoss():
    def __init__(self, logits, labels):
        super(FocalLoss, self).__init__()
        self.logits = logits
        self.labels = labels
        self.gamma = 0
        self.weight = None
        self.size_average = True

    def loss_function(self):
        
        if self.logits.dim() > 2:
            self.logits = self.logits.contiguous().view(self.logits.size(0), self.logits.size(1), -1)
            self.logits = self.logits.transpose(1, 2)
            self.logits = self.logits.contiguous().view(-1, self.logits.size(2)).squeeze()
        if self.labels.dim() == 4:
            self.labels = self.labels.contiguous().view(self.labels.size(0), self.labels.size(1), -1)
            self.labels = self.labels.transpose(1, 2)
            self.labels = self.labels.contiguous().view(-1, self.labels.size(2)).squeeze()
        elif self.labels.dim() == 3:
            self.labels = self.labels.view(-1)
        else:
            self.labels = self.labels.view(-1, 1)
        
        weight = Variable(self.weight)
        logpt = -F.cross_entropy(self.logits, self.labels)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
class BinaryCrossEntropy():
    def __init__(self, logits, labels):
        super(BinaryCrossEntropy, self).__init__()
        self.logits = logits
        self.labels =  labels
    def loss_function(self):
        return F.binary_cross_entropy_with_logits(self.logits,self.labels)

class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).to("cuda:0").scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.sigmoid(input)
        target = target.float()

        # Numerator Product
        inter = (pred * target)
        

        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,  -1).sum(1)

        # Denominator
        union = pred + target - (pred * target)
         # Sum over all pixels N x C x H x W => N x C
        union = union.view(N,  -1).sum(1)

        IOU = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return 1 - IOU.mean()