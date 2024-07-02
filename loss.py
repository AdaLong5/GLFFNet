import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as F

def sim(x,y):
    return (x - y).pow(2).mean(1)

def IAML_loss(Z1, Z2, cls_list, margin=5.0):
    '''
    Z1 = Z2 (both are same features)
    Z1, Z2 could be (sketch, model)
    '''
    total_loss = 0
    dist_table = torch.cdist(Z1, Z2, p=2)
    positive_idx = list()
    negative_idx = list()
    for i, c in enumerate(cls_list.tolist()):
      
        positive_idx = [index for index, el in enumerate(cls_list.tolist()) if el == c]
        positive_dist_table = dist_table[i, positive_idx]
        positive_dist = torch.mean(positive_dist_table)

        negative_idx = [index for index, el in enumerate(cls_list.tolist()) if el != c]
        negative_dist_table = dist_table[i, negative_idx]
        negative_dist = torch.mean(negative_dist_table)

        loss =  positive_dist + F.relu(margin - negative_dist)
        total_loss += loss
    total_loss = total_loss / len(cls_list.tolist())
    return total_loss


class CompactCenterLoss(nn.Module):
    def __init__(self, margin=0, num_classes=10,feat_dim=256):
        super(CompactCenterLoss, self).__init__() 
        self.margin = margin 
        self.ranking_loss = nn.MarginRankingLoss(margin=margin) 
        self.centers = nn.Parameter(torch.randn(num_classes,feat_dim)) 
     
   
    def forward(self, inputs, targets): 
        batch_size = inputs.size(0) 
        targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.size(1)) 
        
        
        centers_batch = self.centers.gather(0, targets_expand) # centers batch 

        # compute pairwise distances between input features and corresponding centers 
        centers_batch_bz = torch.stack([centers_batch]*batch_size) 
        inputs_bz = torch.stack([inputs]*batch_size).transpose(0, 1) 
        dist = torch.sum((centers_batch_bz -inputs_bz)**2, 2).squeeze() 
        dist = dist.clamp(min=1e-12).sqrt() # for numerical stability 

        # for each anchor, find the hardest positive and negative 
        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], [] 
        for i in range(batch_size): # for each sample, we compute distance 
            dist_ap.append(dist[i][mask[i]].max()) 
            dist_an.append(dist[i][mask[i]==0].min())  
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

      
        y = dist_an.data.new() 
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        # y_i = 1, means dist_an > dist_ap + margin will casuse loss be zero 
        loss_1 = self.ranking_loss(dist_an, dist_ap, y)
        loss_2 = IAML_loss(inputs,inputs,targets)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)  

        loss = loss_1 + loss_2 * 0.5
        

        return loss, prec 
