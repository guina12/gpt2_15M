from batch_config import  calc_loss_batch_loader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch 

#-------------------------------------------------------------------------------------------------------------------------------------

def evaluate_model(model, train_loader, val_loader, num_batches, device):
    model.eval()
    with torch.no_grad():
        loss_train = calc_loss_batch_loader(model, train_loader, num_batches , device)
        loss_val = calc_loss_batch_loader(model , val_loader, num_batches, device)

    model.train()

    return loss_train, loss_val

 # ------------------------------------------------------------------------------------------------------------------------------------

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(epochs_seen, train_losses, label="Training loss", 
             color='#2E86AB', linewidth=2)
    ax1.plot(epochs_seen, val_losses, label="Validation loss", 
             linestyle='--', color='#A23B72', linewidth=2)
    
    ax1.set_xlabel("Epochs", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Loss", fontsize=12, fontweight='bold')
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen", fontsize=12, fontweight='bold')
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax2.set_xlim(tokens_seen[0], tokens_seen[-1])
    
    plt.title("Training and Validation Loss", fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    plt.show()