from load_file import  save_openwebtext_to_txt
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data  import Dataset, DataLoader
from train_model import train
from torch.functional import F


GPT_CONFIG_124M = {
    "vocab_size": 50304,
    "context_size": 32,   
    "emb_dim": 32, 
    "n_heads": 4,          
    "n_layers": 4,         
    "drop_rate": 0.7,     
    "qkv_bias": True        
}


MAX_LINES = 10_000_000
raw_text = save_openwebtext_to_txt() 
raw_text = raw_text[ : MAX_LINES]
print(len(raw_text))
tokenizer = tiktoken.encoding_for_model("gpt2")
ration = 0.8
percent = int(len(raw_text) * ration)
train_data_text = raw_text[:percent]
test_data_text = raw_text[percent:]

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#

class TokenizeDataset(Dataset):
    def __init__(self, data ,tokenizer, context_size, stride):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.stride = stride
        self.encoded_text = tokenizer.encode(data)
       
        self.input_ids  = []
        self.target_ids = []

        for i in range(0, len(self.encoded_text) - self.context_size, self.stride):
            x = self.encoded_text[i : i + self.context_size]
            y = self.encoded_text[i + 1 : i + self.context_size + 1]
            self.input_ids.append(torch.tensor(x))
            self.target_ids.append(torch.tensor(y))

    

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    
    def __len__(self):
        return len(self.input_ids)
    

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def create_data_loaders(
    data, tokenizer, 
    context_size, stride,
    batch_size, 
    shuffle = False
):
    dataset = TokenizeDataset(data, tokenizer, context_size, stride)
    loader = DataLoader(
        dataset,
        shuffle    = shuffle,
        batch_size = batch_size,
        num_workers  = 0
    )

    print(f'loaded:{len(loader)}')
    print(f'1 Epoch = {len(loader) // context_size * batch_size}  batches')

    return loader

train_loader = create_data_loaders(train_data_text, tokenizer,  GPT_CONFIG_124M["context_size"], 1 ,  shuffle  = True,   batch_size  = 256)
test_loader  = create_data_loaders(test_data_text, tokenizer,   GPT_CONFIG_124M["context_size"], 1 ,  shuffle  = True,   batch_size  = 256)


class CausualAttention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        
        assert  cfg['emb_dim'] % cfg['n_heads'] == 0

        self.n_head    = cfg['n_heads']
        self.drop_rate = nn.Dropout(cfg['drop_rate'])
        self.wq = nn.Linear(cfg['emb_dim'], cfg['emb_dim'],  bias = cfg['qkv_bias'])
        self.wk = nn.Linear(cfg['emb_dim'], cfg['emb_dim'],  bias = cfg['qkv_bias'])
        self.wv = nn.Linear(cfg['emb_dim'], cfg['emb_dim'],  bias = cfg['qkv_bias'])
        self.cproj = nn.Linear(cfg['emb_dim'], cfg["emb_dim"],bias = cfg['qkv_bias'])
        self.cproj.NANO_GPT_SCALE_INIT = 1

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg['context_size'], cfg['context_size']))
        )
      
    def forward(self, x):
        B, T, C  = x.shape  # x->(B, T, C)

        query = self.wq(x).view(B,  T, self.n_head, C // self.n_head).transpose(1, 2)  #(B,T,C)  -> (B, num_heads, T, head_size)
        key   = self.wk(x).view(B,  T, self.n_head, C // self.n_head).transpose(1, 2)  #(B,T,C)  -> (B, num_heads, T, head_size)
        value = self.wv(x).view(B,  T, self.n_head, C // self.n_head).transpose(1, 2)  #(B,T,C)  -> (B, num_heads, T, head_size)

        #att = query @ key.transpose(2, 3)    #(B, num_heads, T , head_size) @ (B, num_heads, head_size, T) -> (B, num_heads, T, T)
        #attn_masked = att.masked_fill(self.mask[:T, :T] == 0 , float('-inf'))
        #att_score = torch.softmax(attn_masked / key.shape[-1] ** 0.5 , dim = -1)
        #out = att_score @ value     #(B, num_heads, T, T) @ (B, num_heads, T, head_size) -> (B, num_heads, T, head_size)
        out = F.scaled_dot_product_attention(query, key, value, is_causal = True)
        out = out.transpose(1,2).contiguous().view(B, T, C) #(B, T, num_heads , head_size) - > (B,T num_heads * head_size)
        out = self.cproj(out)                #(B, T, C) @ (C,C) -> (B * T, C) @ (C, C) -> (B,T,C)
        out = self.drop_rate(out)

        return out
    

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------#


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg['emb_dim'], cfg['emb_dim']),
            nn.GELU(approximate = 'tanh'),
            nn.Linear(cfg['emb_dim'], cfg['emb_dim']),
            nn.GELU(approximate = 'tanh'),
            nn.Linear(cfg['emb_dim'], cfg['emb_dim']),
            nn.GELU(approximate = 'tanh')
        )
        self.cproj = nn.Linear(cfg['emb_dim'],cfg['emb_dim'])
        self.cproj.NANO_GPT_SCALE_INIT = 1
    
    def forward(self, x):
        out = self.mlp(x)
        return self.cproj(out)
        

#------------------------------------------------------------------------------------------------------------------------------------------------------------------#

class TransformerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg['emb_dim'])
        self.ln2 = nn.LayerNorm(cfg['emb_dim'])
        self.mht = CausualAttention(cfg)
        self.mlp = FeedForward(cfg)
        self.drop_out = nn.Dropout(cfg['drop_rate'])

    
    def forward(self, x):

        shortcut = x 
        x = self.ln1(x)
        x = self.mht(x)
        x = self.drop_out(x)
        x += shortcut

        shortcut = x
        
        x  = self.ln2(x)
        x = self.mlp(x)
        x = self.drop_out(x)
        x += shortcut

        return x
    
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

class GPT124M(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_size"], cfg["emb_dim"])
        self.n_layers = cfg["n_layers"]

        self.trb = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.ln_f   = nn.LayerNorm(cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.lm_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'])

        self.emb.weight = self.lm_head.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if  isinstance(module , nn.Linear):
            std = 0.02
            if hasattr(module,"NANO_GPT_SCALE_INIT"):
                std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0 , std = std)
            if  module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0 , std =  0.02)

    def forward(self, x):
        B, T = x.shape
        emb = self.emb(x)
        pos_emb = self.pos_emb(torch.arange(T,device = x.device))
        x = emb + pos_emb
        x = self.drop_emb(x)
        x = self.trb(x)
        x = self.ln_f(x)
        x = self.lm_head(x)

        return x

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#

torch.set_float32_matmul_precision("high")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT124M(GPT_CONFIG_124M)
model.to(device)  
number_of_parameters = sum(p.numel() for p in model.parameters())


print(f"Params: {number_of_parameters:,}M")

print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")

model.compile()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 3e-4,
    weight_decay = 0.1
)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

loader = iter(train_loader)
x, y   =  next(loader)
sample , y = x.to(device), y.to(device)

train_losses, val_losses, tokens_seens = train(
    model, train_loader,  test_loader ,5 ,5, optimizer, device, 1 ,
    0.7 , 50, GPT_CONFIG_124M['context_size'], 60,
    sample[0].view(1, len(sample[0])), tokenizer
)