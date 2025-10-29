import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader



# Inputs (Long for indices, Float for feature mats)
y = torch.as_tensor(y, dtype=torch.long)
model_idx = torch.as_tensor(model_idx, dtype=torch.long)
dilemma_idx = torch.as_tensor(dilemma_idx, dtype=torch.long)
same_prev = torch.as_tensor(same_prev_mat, dtype=torch.float32)
E_prev = torch.as_tensor(E_prev_mat, dtype=torch.float32)
E_within = torch.as_tensor(E_within_mat, dtype=torch.float32)

N, K = same_prev.shape
M = int(model_idx.max().item()) + 1
D = int(dilemma_idx.max().item()) + 1

class ConformityMAP(nn.Module):
    def __init__(self, M, D, K):
        super().__init__()
        # Unconstrained params; we center over classes in forward for ID
        self.theta_raw = nn.Parameter(torch.zeros(M, K))  # model×class
        self.phi_raw   = nn.Parameter(torch.zeros(D, K))  # dilemma×class
        self.alpha     = nn.Parameter(torch.zeros(M))     # self-stickiness per model
        self.gamma_prev   = nn.Parameter(torch.tensor(0.0))
        self.gamma_within = nn.Parameter(torch.tensor(0.0))

    def forward(self, model_idx, dilemma_idx, same_prev, E_prev, E_within):
        # center across classes to impose sum-to-zero per row
        theta = self.theta_raw - self.theta_raw.mean(dim=1, keepdim=True)  # (M,K)
        phi   = self.phi_raw   - self.phi_raw.mean(dim=1, keepdim=True)    # (D,K)

        th = theta[model_idx]                  # (N,K)
        ph = phi[dilemma_idx]                  # (N,K)
        stick = self.alpha[model_idx].unsqueeze(1) * same_prev     # (N,K)
        infl_prev   = self.gamma_prev   * E_prev                   # (N,K)
        infl_within = self.gamma_within * E_within                 # (N,K)

        logits = th + ph + stick + infl_prev + infl_within         # (N,K)
        return torch.log_softmax(logits, dim=1)                    # (N,K)

model = ConformityMAP(M, D, K)

# MAP objective = negative log-likelihood - log-priors (i.e., + L2 penalties)
# Priors: theta, phi ~ N(0, σ^2), alpha ~ N(0, τ^2), gammas ~ N(0, 0.5^2)
sigma_theta = 1.0
sigma_phi   = 1.0
sigma_alpha = 0.5
sigma_gamma = 0.5

opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)  # weight_decay=0; we’ll add explicit L2

batch = 4096
loader = DataLoader(TensorDataset(model_idx, dilemma_idx, same_prev, E_prev, E_within, y),
                    batch_size=batch, shuffle=True, drop_last=False)

for epoch in range(50):  # tune as needed
    total = 0.0
    for mi, di, sp, ep, ew, yy in loader:
        logp = model(mi, di, sp, ep, ew)              # (B,K)
        nll = nn.NLLLoss(reduction='mean')(logp, yy)  # negative log-likelihood

        # L2 priors (sum of squares / (2σ^2))
        theta = model.theta_raw
        phi   = model.phi_raw
        alpha = model.alpha
        gp = model.gamma_prev
        gw = model.gamma_within

        prior = (theta.pow(2).sum() / (2*sigma_theta**2)
                 + phi.pow(2).sum() / (2*sigma_phi**2)
                 + alpha.pow(2).sum() / (2*sigma_alpha**2)
                 + gp.pow(2) / (2*sigma_gamma**2)
                 + gw.pow(2) / (2*sigma_gamma**2))

        loss = nll + prior / N   # scale prior by N for sensible balance
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * mi.shape[0]
    # print or log loss if you want