#!/usr/bin/env python

#################################
#                               #
#    Custom Loss Functions      #
#    created by Warren Kozak    #
#                               #
#################################

# TODO: Implement DiceLoss and CombinedLoss classes for a combined
#       cross entropy and dice loss function for use when training.

# Imports
import torch

from torch import nn

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, labels):
        predictions = torch.softmax(predictions)
                

class CombinedLoss(nn.Module):

    def __init__(self, cel_coeff, dice_coeff, cel_weights=None):
        super().__init__()

        self.cel_coeff = cel_coeff
        self.dic_coeff = dice_coeff

        self.cel = nn.CrossEntropyLoss(weight=cel_weights)
        self.dice = DiceLoss()

    def forward(self, predictions, labels):
        return