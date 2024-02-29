import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from src.utils import Dict2Class


class CausalAttention(nn.Module):
    """
    
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Causal mask to ensure that attention is only applied to the left in the input sequence
        # We need to .view(1, 1, ...) so the  mask is broadcastable with the shape of the tensor
        # * The tensor will have the shape:   (batch_size, num_heads, seq_len    , seq_len)
        # * The mask needs to have the shape: (         1,         1, block_size , block_size)
        # where block_size is the maximum sequence length; if the sequence is shorter,
        # the mask will be appropriately sliced in the forward() method
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))        
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout_prob)

        
    def forward(self, q, k, v, seq_len):
        # All shapes: (batch_size, num_heads, embed_size, qkv_size)
        
        # Perform q * k^T (* is the dot product here)
        # We have to use @ since we work with batches!
        out = q @ k.transpose(-2, -1)
        
        # Divide by scaling factor
        out = out / (q.shape[-1] ** 0.5)        

        # Apply causal mask by replacing all values with -inf where mask is 0
        # A value of -inf will yield a 0 after applying the softmax
        out = out.masked_fill(self.mask[:,:,:seq_len,:seq_len] == 0, float('-inf'))        
        
        # Push through softmax layer
        out = F.softmax(out, dim=-1)
        
        # Push through dropout lauer
        out = self.dropout(out)
        
        # Multiply with values and return result
        return out @ v
        
        
        

class CausalMultiHeadAttention(nn.Module):
    """
    
    """

    def __init__(self, config):
        super().__init__()
        
        # Check if the embedding size can equally be split across all heads
        assert config.embed_size % config.num_heads == 0
        
        ## Side note: the order of the definition of layers/modules matters
        ## but only so it matches the pretrained models from HuggingFace.
        ## Without it, the loading of pretraind models would require to adhere
        ## to the exact same naming scheme of the HuggingFace models
        
        # Scaled Dot Product Attention Layer
        self.attention = CausalAttention(config)                
        
        # For better performance, we use a single nn.Linear layer
        # for the query, key, value projections and for all heads
        # => this mean we need to split things later in forward()
        self.Wa = nn.Linear(config.embed_size, 3*config.embed_size)
                
        # Linear layer to "unify" all heads into one
        self.Wo = nn.Linear(config.embed_size, config.embed_size)
        
        
        # regularization
        #self.attention_dropout_prob = nn.Dropout(config.attention_dropout_prob)
        self.dropout  = nn.Dropout(config.residual_dropout_prob)
        
        self.num_heads  = config.num_heads
        self.embed_size = config.embed_size
         
        # Define sizes of query/key/value based on embedding size and number of heads
        self.qkv_size = self.embed_size // self.num_heads
        

    def forward(self, x):
        batch_size, seq_len, embed_size = x.size() # batch size, sequence length, embedding size
        
        # Check if self.embed_size == embed_size (otherwise something is wrong!)
        assert self.embed_size == embed_size

        # Push batch through "joint project" for the query, key, and value
        # And the split the output to get the views to the key, query, and value
        q, k, v  = self.Wa(x).split(self.embed_size, dim=2)
        
        # The shape of q, k, and v is now        
        # (batch_size, seq_len, embed_size)
        
        # Create views for the query, key and value to "separate" all heads
        # We also transpose dimensions 1 and 2 as preparation for the matrix multiplications
        q = q.view(batch_size, seq_len, self.num_heads, self.qkv_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.qkv_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.qkv_size).transpose(1, 2)

        # The shapes of q, k, and v are now all:
        # (batch_size, num_heads, seq_len, qkv_size)       

        # Perform self-attention over ALL heads
        out = self.attention(q, k, v, seq_len)
        
        # The shape of out: (batch_size, num_heads, seq_len, qkv_size)
        
        # Re-assemble all head outputs side by side
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size) 

        # The shape of out: (batch_size, seq_len, embed_size)
        
        # Push through ouput projection and return result
        return self.dropout(self.Wo(out))

    

class MLPLayer(nn.Module):
    """
    
    """
    
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_size, config.mlp_factor*config.embed_size),
            nn.GELU(),
            nn.Linear(config.mlp_factor*config.embed_size, config.embed_size),
            nn.Dropout(config.residual_dropout_prob),
        )
        
    def forward(self, x):
        return self.mlp(x)
    
    
    
class TransformerBlock(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(config.embed_size)
        self.mha = CausalMultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_size)
        self.mlp = MLPLayer(config)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x        
        

        
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.block_size = config.block_size
        
        ###################################################################################
        ##
        ## Define all layers
        ##
        ###################################################################################        
        
        self.token_embedding      = nn.Embedding(config.vocab_size, config.embed_size)
        self.positional_embedding = nn.Embedding(config.block_size, config.embed_size)
        
        self.embedding_dropout = nn.Dropout(config.embedding_dropout_prob)
        
        # Define num_layers (N) transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [ TransformerBlock(config) for _ in range(config.num_layers) ]
        )
        
        self.ln = nn.LayerNorm(config.embed_size)
        
        self.lm_head = nn.Linear(config.embed_size, config.vocab_size, bias=False)
        
        ###################################################################################
        ##
        ## Initialize all weights
        ##
        ###################################################################################
        
        # Use the nn.Module.apply method to recursively apply the initialization
        self.apply(self._init_weights)
        
        # Apply a special scaled init to the output projections (cf. GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('Wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))
        
        ###################################################################################
        ##
        ## For reporting only
        ##
        ###################################################################################        
        
        # report number of parameters (note we don't count the decoder parameters in lm_head)
        num_params = 0
        num_params += sum(p.numel() for p in self.token_embedding.parameters())
        num_params += sum(p.numel() for p in self.positional_embedding.parameters())
        num_params += sum(p.numel() for p in self.transformer_blocks.parameters())
        print("number of parameters: %.2fM" % (num_params/1e6,))
        
        

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)        
        
        
        
    def forward(self, inputs, targets=None):
        batch_size, seq_len = inputs.size()
        
        # Check if the sequence length does not exceed the maximum lenght specified by config.block_size
        assert seq_len <= self.config.block_size, f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"
               
        # Compute token and positional encodings
        pos = torch.arange(0, seq_len, dtype=torch.long, device=inputs.device).unsqueeze(0) # shape (1, seq_len)
        pos_embed = self.positional_embedding(pos) # position embeddings of shape (1, seq_len, embed_size)
        tok_embed = self.token_embedding(inputs) # token embeddings of shape (batch_size, seq_len, embed_size)
        
        # Combine token and positional embedding via broadcasting
        # (positional encodings are the same for each sequence!)
        x = self.embedding_dropout(tok_embed + pos_embed)
        
        # Push batch through all num_layers transformer blocks
        for block in self.transformer_blocks:
            x = block(x)        
            
        x = self.ln(x)
        
        logits = self.lm_head(x)
        
        # If targets is not None (i.e., we are training!), also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.config.ignore_index)

        return logits, loss
    
    
    
    
    @torch.no_grad()
    def generate(self, token_indices, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (batch_size, seq_len)) and
        complete the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            if token_indices.size(1) <= self.block_size:
                token_indices_condition = token_indices
            else:
                token_indices_condition = token_indices[:, -self.block_size:]
            #idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Push token indices through model to get the logits for each token in the sequence
            logits, _ = self(token_indices_condition)
            
            # Get the logic at the final step and scale by desired temperature;
            # recall that we get an ouput (logit) for all of the tokens, but we
            # are only interested in the last one since this will be the new token
            logits = logits[:, -1, :] / temperature
            
            # Optional: crop the logits to only the top k options to restrict the possible 
            # tokens to some subset of the most likely tokens in case do_sample=True
            # (if do_sample=Fasel, we pick the most likely one anyway)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Either sample from the distribution or take the most likely element
            if do_sample:
                token_index_next = torch.multinomial(probs, num_samples=1)
            else:
                _, token_index_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            token_indices = torch.cat((token_indices, token_index_next), dim=1)

        # Return complete list of token indices (initial list + all generated tokens)
        return token_indices    
    
    
    
    
    
    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        #assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2'}, f"Unknown model_type '{model_type}'"
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2'}, f"Unknown model_type '{model_type}'"
        from transformers import GPT2LMHeadModel
    
        # create a from-scratch initialized minGPT model
        config = Dict2Class(gpt_base_configs[model_type])
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        
        ###################################################################################
        # Create "local" GPT model
        model = GPT(config)
        sd = model.state_dict()    
        
        ###################################################################################
        # Load and initialize GPT model from Huggingface
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

    
    
        # Get keys (i.e., the names of all layers/modules) of HuggingFace GPT model)
        keys_hf = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        # Get keys (i.e., the names of all layers/modules) of local GPT model)
        keys = [k for k in sd]
    
        # Without the 'attn.masked_bias' layers/modules, the number of keys must match
        assert len(keys_hf) == len(keys)
        
        ## Side note: Our implementation uses different names for the layers/modules.
        ## This is not a problem as long as the order of layers/modules of our implementation
        ## matches the order of the HuggingFace implementation
        
    
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        for key, key_hf in zip(keys, keys_hf):
            if any(key_hf.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[key_hf].shape[::-1] == sd[key].shape
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key_hf].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[key_hf].shape == sd[key].shape, f"{key_hf}: {sd_hf[key_hf].shape} vs. {key}: {sd[key].shape}"
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key_hf])

        return model        
    
    
    
    
    
gpt_base_configs = {
    'gpt2': {
        'model_type': 'gpt2',
        'vocab_size': None,
        'embed_size': 768,
        'num_heads': 12,
        'num_layers': 12,
        'block_size': 1024,
        'embedding_dropout_prob': 0.1,
        'attention_dropout_prob': 0.1,
        'residual_dropout_prob': 0.1,
        'mlp_factor': 4,
        'ignore_index': -1             
    },
    'gpt2-medium': {
        'model_type': 'gpt2-medium',
        'vocab_size': None,
        'embed_size': 1024,
        'num_heads': 16,
        'num_layers': 24,
        'block_size': 1024,
        'embedding_dropout_prob': 0.1,
        'attention_dropout_prob': 0.1,
        'residual_dropout_prob': 0.1,
        'mlp_factor': 4,
        'ignore_index': -1             
    },
    'gpt2-large': {
        'model_type': 'gpt2-large',
        'vocab_size': None,
        'embed_size': 1280,
        'num_heads': 20,
        'num_layers': 36,
        'block_size': 1024,
        'embedding_dropout_prob': 0.1,
        'attention_dropout_prob': 0.1,
        'residual_dropout_prob': 0.1,
        'mlp_factor': 4,
        'ignore_index': -1             
    },    
    'gpt2-xl': {
        'model_type': 'gpt2-xl',
        'vocab_size': None,
        'embed_size': 1600,
        'num_heads': 25,
        'num_layers': 48,
        'block_size': 1024,
        'embedding_dropout_prob': 0.1,
        'attention_dropout_prob': 0.1,
        'residual_dropout_prob': 0.1,
        'mlp_factor': 4,
        'ignore_index': -1             
    },  
    'distilgpt2': {
        'model_type': 'distilgpt2',
        'vocab_size': None,
        'embed_size': 768,
        'num_heads': 12,
        'num_layers': 6,
        'block_size': 1024,
        'embedding_dropout_prob': 0.1,
        'attention_dropout_prob': 0.1,
        'residual_dropout_prob': 0.1,
        'mlp_factor': 4,
        'ignore_index': -1             
    },       
    'gpt-tiny': {
        'model_type': 'gpt-tiny',
        'vocab_size': None,
        'embed_size': 32,
        'num_heads': 4,
        'num_layers': 3,
        'block_size': 64,
        'embedding_dropout_prob': 0.1,
        'attention_dropout_prob': 0.1,
        'residual_dropout_prob': 0.1,
        'mlp_factor': 4,
        'ignore_index': -1    
    }
}