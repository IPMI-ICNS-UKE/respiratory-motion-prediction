from rmp.my_utils.logger import LoggerMixin


class EarlyStopping(LoggerMixin):
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0, phase: str = ""):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.phase = phase

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif (self.best_loss - val_loss) / self.best_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif (self.best_loss - val_loss) / self.best_loss < self.min_delta:
            self.counter += 1
            self.logger.info(
                f"Early stopping {self.phase} counter {self.counter} of {self.patience}"
            )
            if self.counter >= self.patience:
                self.logger.warning(f"Early {self.phase} stopping activated!")
                self.early_stop = True
