import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm
import pandas as pd

# =============================================================================
# 1. Architecture (Symmetry-Locked Layer + Dobrushin Transformer)
# =============================================================================

class SpectralLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.proj = spectral_norm(nn.Linear(in_features, out_features, bias=bias))
    def forward(self, x):
        return self.proj(x)

class BandControlledAttention(nn.Module):
    def __init__(self, d_model, n_heads, R=16):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.R = R
        self.q_proj = SpectralLinear(d_model, d_model)
        self.k_proj = SpectralLinear(d_model, d_model)
        self.v_proj = SpectralLinear(d_model, d_model)
        self.out_proj = SpectralLinear(d_model, d_model)
        self.log_lambda = nn.Parameter(torch.tensor(math.log(math.exp(0.8) - 1.0)))

    def forward(self, x, memory_k, memory_v):
        B, T, C = x.size()
        M = memory_k.size(2)
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        memory_k_exp = memory_k.expand(B, -1, -1, -1)
        memory_v_exp = memory_v.expand(B, -1, -1, -1)
        k = torch.cat([memory_k_exp, k], dim=2)
        v = torch.cat([memory_v_exp, v], dim=2)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        memory_mask = torch.ones(1, 1, T, M, device=x.device)
        full_mask = torch.cat([memory_mask, causal_mask], dim=3)
        scores = scores.masked_fill(full_mask == 0, float('-inf'))
        
        lam = F.softplus(self.log_lambda)
        positions = torch.arange(T, device=x.device).unsqueeze(1)
        history_positions = torch.cat([torch.zeros(M, device=x.device)-1, torch.arange(T, device=x.device)])
        dist = torch.abs(positions - history_positions.unsqueeze(0))
        penalty = lam * torch.clamp(dist - self.R, min=0.0)
        
        scores = scores - penalty.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), lam

class ContractiveBlock(nn.Module):
    def __init__(self, d_model, n_heads, gamma=0.05):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = BandControlledAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            SpectralLinear(d_model, 4 * d_model),
            nn.GELU(),
            SpectralLinear(4 * d_model, d_model)
        )
        self.gamma = gamma

    def forward(self, x, memory_k, memory_v):
        attn_out, lam = self.attn(self.ln_1(x), memory_k, memory_v)
        x = x + self.gamma * attn_out
        x = x + self.gamma * self.mlp(self.ln_2(x))
        return x, lam

class SymmetryLockedLayer(nn.Module):
    def __init__(self, d_model, vocab_size, k_centers_csv, tau=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tau = tau 
        
        df = pd.read_csv(k_centers_csv)
        self.K = len(df)
        self.cluster_centers = nn.Parameter(torch.randn(self.K, d_model))
        self.transition_operator = nn.Parameter(torch.randn(self.K, vocab_size))
        
    def forward(self, h):
        B, T, D = h.size()
        h_flat = h.view(B * T, D)
        dists = torch.cdist(h_flat, self.cluster_centers, p=2)
        symbol_probs = F.softmax(-dists / self.tau, dim=-1)
        logits = torch.matmul(symbol_probs, self.transition_operator)
        return logits.view(B, T, self.vocab_size), symbol_probs.view(B, T, self.K)

class SLL_MathNet_Transformer(nn.Module):
    def __init__(self, vocab_size=256, d_model=512, n_heads=8, n_layers=6, M=16, symbols_csv="mathnet_extracted_symbols.csv"):
        super().__init__()
        self.M = M
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(1024, d_model)
        
        self.memory_k = nn.Parameter(torch.randn(1, n_heads, M, d_model // n_heads))
        self.memory_v = nn.Parameter(torch.randn(1, n_heads, M, d_model // n_heads))
        
        self.blocks = nn.ModuleList([
            ContractiveBlock(d_model, n_heads, gamma=0.5/(layer+1)**1.5) 
            for layer in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.sll = SymmetryLockedLayer(d_model, vocab_size, symbols_csv)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        lambdas = []
        for block in self.blocks:
            x, lam = block(x, self.memory_k, self.memory_v)
            lambdas.append(lam)
            
        hidden = self.ln_f(x)
        logits, sym_probs = self.sll(hidden)
        
        return logits, sym_probs

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=True, top_k=None):
        """
        Rollout generation using Symmetry-Locked transition constraints.
        """
        self.eval()
        symbol_trace = []
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]
            
            logits, sym_probs = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            sym_state = torch.argmax(sym_probs[:, -1, :], dim=-1).item()
            symbol_trace.append(sym_state)
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
                
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx, symbol_trace

# =============================================================================
# 2. Execution & Generation Runner
# =============================================================================

def decode_bytes(tensor):
    tokens = tensor.squeeze().tolist()
    return "".join([chr(b) if b < 128 else "" for b in tokens])

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    symbols_csv = "mathnet_extracted_symbols.csv"
    ckpt_path = "mathnet_e2e_checkpoint_15000.pt" 
    
    print(f"Loading End-to-End SLL Model from {ckpt_path}...")
    model = SLL_MathNet_Transformer(symbols_csv=symbols_csv).to(device)
    
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print("End-to-End Weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Missing {ckpt_path}! Make sure the e2e training script successfully saved the checkpoint.")

    prompts = [
        "Problem:\nLet $x$ be a real number such that",
        "Solution:\nWe can solve this by factoring the polynomial",
        "Consider the set $A = \\{ 1, 2,"
    ]
    
    print("\n============================================================")
    print("SLL END-TO-END ROLLOUT EVALUATION")
    print("============================================================\n")
    
    for p in prompts:
        print(f"PROMPT: {p}")
        context = torch.tensor(list(p.encode('utf-8')), dtype=torch.long, device=device).unsqueeze(0)
        
        out_idx, sym_trace = model.generate(context, max_new_tokens=100, temperature=0.8, do_sample=True, top_k=5)
        sym_path = " -> ".join([str(s) for s in sym_trace[:15]]) + " ..."
        
        print(f"GENERATION:\n{decode_bytes(out_idx)}\n")
        print(f"Invariant Path Traversed: {sym_path}")
        print("-" * 60)

if __name__ == "__main__":
    main()
