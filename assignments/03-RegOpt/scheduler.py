"""
Customed Learning Rate Scheduler, 
inheriting torch.optim.lr_scheduler._LRScheduler
"""
from typing import List
import math

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    A customed learning rate scheduler
    inheriting torch.optim.lr_scheduler._LRScheduler
    """

    def __init__(
        self, optimizer, last_epoch=-1, start_batch=7820
    ):  # lr unchanged for 15 epochs
        """
        Create a new scheduler.
        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.
        """
        # ... Your Code Here ...
        self.update_count = 0  # update lr per 782 batches
        self.start_batch = start_batch
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Getting current learning rate
        This scheduler will update learning rate each 782 iterations
        (one epoch for batchsize 64)
        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...

        if self.last_epoch < self.start_batch:
            # print("using base lr: ", self.base_lrs)
            return [i for i in self.base_lrs]
        else:
            self.update_count += 1
            if self.update_count == 782:
                self.update_count = 0  # reset to 0
                return [
                    group["lr"] / (0.5 * math.log(0.5 * self.last_epoch) ** 0.5)
                    for group in self.optimizer.param_groups
                ]
            else:
                return self.get_last_lr()
