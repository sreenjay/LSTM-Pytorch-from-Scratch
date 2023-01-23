import torch


# Learning rate scheduler

class LrScheduler:
    
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        """
        new_lr = old_lr * factor
        
        Parameters:
        ------------
        optimizer : the optimzer the function is using
        patience  : how many epochs to wait before updating learning rate
        min_lr    : the least value the learning rate can take
        factor    : factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=self.patience, 
                                                                       factor=self.factor, min_lr=self.min_lr, verbose=True)
        
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
        
        
    


# Early Stopping

class EarlyStopping:
    
    def __init__(self, patience=5, min_delta=0, factor=0.7):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter: {self.counter} of patience: {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                
                
                
                
# Save Best Model
class SaveBestModel:
    """
    Class to save the best model while training.
        - If the current epoch's validation loss is less than the
        previous least validation loss then save the model.
    """
    def __init__(self, best_val_loss=float('inf')):
        self.best_val_loss = best_val_loss
    
    def __call__(self, curr_val_loss, epoch, model, optimizer, criterion, file_name=''):
        if curr_val_loss < self.best_val_loss:
            self.best_val_loss = curr_val_loss
            print(f"\nBest validation loss: {self.best_val_loss:.3f}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({'epoch':epoch+1, 
                        'model_state_dict'     : model.state_dict(), 
                        'optimizer_state_dict' : optimizer.state_dict(), 
                        'loss':criterion}, f"best_model{file_name}.pth")