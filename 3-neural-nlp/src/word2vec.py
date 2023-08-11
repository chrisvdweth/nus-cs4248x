import torch.nn as nn


class CBOW(nn.Module):
    
    def __init__(self, vocab_size, embed_dim):
        super(CBOW, self).__init__()
        self.V = nn.Embedding(vocab_size, embed_dim, max_norm=1)
        self.U = nn.Linear(embed_dim, vocab_size)

    def forward(self, contexts):
        out = self.V(contexts)
        out = out.mean(axis=1)
        out = self.U(out)
        return out
    
    
class Skipgram(nn.Module):
    
    def __init__(self, vocab_size, embed_dim):
        super(Skipgram, self).__init__()
        self.V = nn.Embedding(vocab_size, embed_dim, max_norm=1)
        self.U = nn.Linear(embed_dim, vocab_size)

    def forward(self, centers):
        out = self.V(centers)
        out = self.U(out)
        return out
