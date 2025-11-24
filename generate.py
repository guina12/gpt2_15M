import torch.nn as nn 
import torch 

def generate_text(
        model, idx, 
        context_size, max_new_tokens, 
        temperature,top_k = None
    ):
    
    for  i in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        model.eval()
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:,-1,:]

        if top_k is not None and temperature > 0.0:
            logits_top_k = torch.topk(logits, k = top_k)
            min_value = logits_top_k.values[:,-1]
            logits = torch.where(
                condition = min_value > logits,
                input = torch.tensor(float('-inf')),
                other =  logits
            )

            logits = logits / temperature
            probs = torch.softmax(logits, dim = -1 )
            next_token = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, next_token), dim = -1 )
        else:
            probs = torch.softmax(logits, dim = -1)
            next_token = torch.argmax(probs, keepdim = True)
            idx = torch.cat((idx,next_token), dim = 1)
    
    return  idx
    
            
