import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_score, neg_score, sum=True):
        # pos_score: B | B x L | B x L
        # neg_score: B x neg | B x L x neg | B x neg
        
        weight = self._cal_weight(neg_score)
        padding_mask = torch.isinf(pos_score)
        # positive
        pos_loss = F.logsigmoid(pos_score)
        pos_loss.masked_fill_(padding_mask, 0.0)
        if sum:
            pos_loss = pos_loss.sum() / (~padding_mask).sum()
        else:
            pos_loss = pos_loss / (~padding_mask).sum()
        # negative
        neg_loss = F.softplus(neg_score) * weight
        neg_loss = neg_loss.sum(-1)
        # mask
        if pos_score.dim() == neg_score.dim()-1:
            neg_loss.masked_fill_(padding_mask, 0.0)
            if sum:
                neg_loss = neg_loss.sum() / (~padding_mask).sum()
            else:
                neg_loss = neg_loss / (~padding_mask).sum()
        else:
            neg_loss = torch.mean(neg_loss)
        # return -pos_loss
        return -pos_loss + neg_loss

    def _cal_weight(self, neg_score):
        return torch.ones_like(neg_score) / neg_score.size(-1)
    
class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_score, neg_score):
        padding_mask = torch.isinf(pos_score)
        loss = F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score)
        loss.masked_fill_(padding_mask.unsqueeze(-1), 0.0)
        weight = F.softmax(torch.ones_like(neg_score), -1)
        return -(loss * weight).sum(-1).sum() / (~padding_mask).sum()