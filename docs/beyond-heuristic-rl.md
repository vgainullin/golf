# Improving Beyond the Heuristic: RL Approaches

## Problem Statement

A DQN agent can perfectly learn a Golf card game heuristic via imitation learning (100% accuracy, identical game performance at ~14.0 per hole). But RL training (DQN with experience replay, tournament population training) fails to improve beyond the heuristic -- agents either plateau or degrade.

The heuristic is far from optimal: simply allowing it to replace revealed cards (not just unrevealed) drops the score from 14.0 to 9.6, a 4.4 point improvement. So there's significant room for a learned policy to improve.

### Why DQN + epsilon-greedy fails

1. **Random exploration can't discover strategic improvements.** The probability of randomly stumbling into "replace a revealed 9 with a drawn 5" is negligible -- it requires a specific non-heuristic action in a specific situation.
2. **DQN has no anchor to the prior.** Q-values drift freely during training, so the agent either collapses or the RL signal is too weak to overcome noise.

### Game characteristics

- Discrete action space (16 actions, ~2-12 valid per state)
- Partial observability (own hidden cards unknown, opponents' hidden cards unknown)
- Stochastic environment (random deck shuffles, random opponents)
- ~72 decision points per 9-hole game per player
- Fast vectorized simulator (10k games in seconds)

## Approaches

### 1. Residual Q-Learning

**Effort:** ~1 hour | **Payoff:** Medium

Decompose Q as `Q_total = Q_base(frozen) + alpha * Q_residual(learned)`. The frozen base is the imitation-learned model. The residual network (same architecture, initialized near zero) only needs to learn the *delta* between heuristic and optimal policy -- zero in most states, non-zero only where the heuristic is suboptimal.

```python
class ResidualDQN(nn.Module):
    def __init__(self, base_model, residual_model, alpha=0.1):
        super().__init__()
        self.base = base_model  # frozen
        self.residual = residual_model  # learned
        self.alpha = alpha

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, state_tokens, stages):
        with torch.no_grad():
            q_base = self.base(state_tokens, stages)
        q_residual = self.residual(state_tokens, stages)
        return q_base + self.alpha * q_residual
```

Start alpha at 0.1, increase to 1.0 over training. Standard DQN training on Q_total, only residual parameters have gradients.

**References:**
- [Residual Q-Learning (2023)](https://arxiv.org/html/2306.09526v3)
- [Residual Policy Learning (Silver et al., 2018)](https://www.researchgate.net/publication/329734796_Residual_Policy_Learning)

### 2. DQfD (Deep Q-Learning from Demonstrations)

**Effort:** ~2-3 hours | **Payoff:** High

Add a large-margin classification loss on demonstration transitions. This forces the expert action's Q-value to stay above alternatives by a margin, preventing catastrophic forgetting while allowing RL to discover improvements.

**Loss function:**

```python
# Total loss = J_DQN + lambda_1 * J_nstep + lambda_2 * J_margin + lambda_3 * J_L2

def margin_loss(q_values, expert_actions, margin=0.8):
    margins = torch.full_like(q_values, margin)
    margins.scatter_(1, expert_actions.unsqueeze(1), 0.0)
    augmented = q_values + margins
    max_q = augmented.max(dim=1).values
    expert_q = q_values.gather(1, expert_actions.unsqueeze(1)).squeeze(1)
    return (max_q - expert_q).mean()
```

**Implementation:**
- Maintain a demo buffer filled by running the vectorized heuristic
- Mixed batches: 75% agent replay buffer, 25% demo buffer
- Margin loss applied only to demo-buffer samples
- Add n-step returns (n=10) for longer-horizon credit assignment
- Demo ratio decays from 25% to 10% over training

**Hyperparameters:**
- lambda_1 (n-step): 1.0
- lambda_2 (margin): 1.0, decay to 0.1
- lambda_3 (L2): 1e-5
- Margin: 0.8

**References:**
- [Deep Q-learning from Demonstrations (Hester et al., 2017)](https://ar5iv.labs.arxiv.org/html/1704.03732)
- [DI-engine DQfD docs](https://di-engine-docs.readthedocs.io/en/latest/12_policies/dqfd.html)

### 3. Targeted Exploration

**Effort:** ~1-2 hours | **Payoff:** Medium

Replace epsilon-greedy with Boltzmann exploration biased toward heuristic deviations. Higher temperature in states where the heuristic advantage is small (more to discover), lower temperature where it's large (exploit known-good behavior).

```python
def adaptive_exploration(q_values, heuristic_action, valid_mask, base_temp=0.5):
    masked_q = q_values.masked_fill(~valid_mask, float('-inf'))
    heuristic_q = q_values.gather(1, heuristic_action.unsqueeze(1)).squeeze(1)
    max_other_q = masked_q.clone()
    max_other_q.scatter_(1, heuristic_action.unsqueeze(1), float('-inf'))
    max_other_q = max_other_q.max(dim=1).values

    gap = (heuristic_q - max_other_q).clamp(min=0)
    temperature = base_temp + 1.0 / (1.0 + gap)

    probs = F.softmax(masked_q / temperature.unsqueeze(1), dim=1)
    return torch.multinomial(probs, 1).squeeze(1)
```

Optionally add an intrinsic reward bonus for taking non-heuristic actions in novel states (count-based exploration).

**References:**
- [Exploration Strategies in Deep RL (Lil'Log)](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/)

### 4. KL-Regularized Policy Gradient (PPO)

**Effort:** ~4-6 hours | **Payoff:** High

Same approach as RLHF: use the imitation policy as a reference, optimize reward with a KL penalty. Requires switching from DQN to actor-critic.

```python
# Objective: max E[ sum_t ( r_t - beta * KL(pi_theta || pi_ref) ) ]
# Modified reward: r_modified = r_original - beta * (log pi_theta(a|s) - log pi_ref(a|s))

def kl_penalty(logits_theta, logits_ref, actions):
    log_pi = F.log_softmax(logits_theta, dim=-1)
    log_ref = F.log_softmax(logits_ref, dim=-1)
    kl = log_pi.gather(1, actions.unsqueeze(1)).squeeze(1) - \
         log_ref.gather(1, actions.unsqueeze(1)).squeeze(1)
    return kl
```

**Details:**
- Beta scheduling: 0.5 (constrained) -> 0.01 (free) over training
- Reference policy: frozen imitation model, or deterministic heuristic with soft labels (0.95 for heuristic action, uniform over rest)
- Entropy bonus alpha=0.01
- Two-head architecture: shared encoder, policy head + value head
- GAE lambda=0.95, gamma=0.99, PPO clip=0.2

**References:**
- [RLHF PPO Implementation Details (HuggingFace)](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)
- [PPO (Spinning Up)](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

### 5. ISMCTS + Expert Iteration

**Effort:** ~8-12 hours | **Payoff:** Very High (best ceiling)

Monte Carlo tree search with information-set sampling. Sidesteps the exploration problem entirely by using search instead of random noise to discover improvements.

1. At each decision, sample K determinizations of hidden information
2. Run MCTS from current state using neural net as rollout policy + value estimator
3. Aggregate action values across determinizations
4. Train neural net to predict MCTS action distribution (Expert Iteration loop)

**Key advantage:** MCTS discovers that replacing a revealed 9 with a drawn 5 is good because it simulates the resulting game. No random exploration needed.

**Cost:** ~2000 forward passes per decision (100 sims x 20 determinizations). At 72 decisions/game, feasible but slower than pure DQN.

**References:**
- [ISMCTS (AI Factory)](https://www.aifactory.co.uk/newsletter/2013_01_reduce_burden.htm)
- [AlphaZero for imperfect information games](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1014561/full)
- [Transformer planning for trick-taking card games](https://arxiv.org/html/2404.13150v1)

### 6. AWR (Advantage-Weighted Regression)

**Effort:** ~3-4 hours | **Payoff:** Medium

Collect trajectories with 80% heuristic / 20% random actions. Estimate advantages via a learned value function. Train policy via weighted maximum likelihood -- good deviations get upweighted.

```python
def awr_loss(logits, actions, advantages, temperature=1.0):
    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    weights = torch.exp(advantages / temperature).detach()
    weights = weights / weights.sum()
    return -(weights * selected_log_probs).sum()
```

**References:**
- [AWR: Simple and Scalable Off-Policy RL](https://openreview.net/pdf/ec69fdc5cafd6a55f98afb0ffea7d424eaee6034.pdf)

## Recommendation

Start with **Residual Q-Learning + DQfD combined**:
- Residual Q-Learning: wrap imitation model as frozen base, add learned residual
- DQfD margin loss: prevents forgetting via demo buffer + margin
- Both address the two root causes (no anchor + bad exploration)

If that doesn't break through, **ISMCTS + Expert Iteration** has the highest ceiling for imperfect-information card games and sidesteps exploration entirely via search.
