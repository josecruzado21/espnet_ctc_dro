import logging
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked
from torch import Tensor
import pdb 

class DROCTCLoss(torch.nn.modules.loss._Loss):
    def __init__(self, blank=0, zero_infinity=False):
        super().__init__()
        self.blank = blank
        self.zero_infinity = zero_infinity

    def forward(self, 
                log_probs: Tensor, 
                targets: Tensor, 
                input_lengths: Tensor, 
                target_lengths: Tensor,
                groups: List[str],
                groups_weights: Dict,
                valid: bool = True) -> Tensor:
        log_probs = log_probs.permute(1, 0, 2)
        losses = F.ctc_loss(
            log_probs, 
            targets, input_lengths, target_lengths, 
            self.blank, reduction='none',
            zero_infinity=self.zero_infinity)
        if not valid:    
            print("DROCTC losses before weighting:", losses)
            print("groups:", groups)
            losses = torch.stack([
                groups_weights[groups[ix]] * losses[ix] for ix in range(losses.shape[0])
            ])
            print("DROCTC losses after weighting:", losses)
        else:
            print("groups:", groups)
            losses = torch.stack([
                losses[ix] for ix in range(losses.shape[0])
            ])
        return losses