import torch.nn as nn
import torch

# ---------------------------------------------------------------------------------------------------------------------------------------------#

def text_to_token_ids(text, tokenizer):
    tokens = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    return tokens

# ---------------------------------------------------------------------------------------------------------------------------------------------#

def token_ids_to_text(token_ids, tokenizer):
    text = tokenizer.decode(token_ids.flatten().tolist())
    return text

# ---------------------------------------------------------------------------------------------------------------------------------------------#

def calc_loss_batch(model , xb, yb, device):
    xb, yb  = xb.to(device), yb.to(device)
    with torch.autocast(device_type = 'cuda' , dtype = torch.bfloat16) : #MIXED-PRECISION
        logits =  model(xb)
    loss = nn.functional.cross_entropy(
        logits.view(logits.shape[0] * logits.shape[1], logits.shape[2]),
        yb.flatten()
    )

    return loss

# ---------------------------------------------------------------------------------------------------------------------------------------------#
   
def calc_loss_batch_loader(model, data_loader, num_batches, device):
    total_loss = 0 

    if num_batches  == 0 or num_batches > len(data_loader):
        num_batches = len(data_loader)
    else:
        for i , (xb, yb) in enumerate(data_loader):
            if i  <= num_batches:
                loss = calc_loss_batch(model, xb, yb , device)
                total_loss += loss.item()
            else:
                break
    
    return total_loss / num_batches

# ---------------------------------------------------------------------------------------------------------------------------------------------#