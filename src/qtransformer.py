import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class CardEmbedding(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        # 52 cards + 1 for unknown/hidden card
        self.card_embedding = nn.Embedding(53, embedding_dim)
        
        # Suit and rank embeddings for better card representation
        self.suit_embedding = nn.Embedding(4, embedding_dim // 2)
        self.rank_embedding = nn.Embedding(13, embedding_dim // 2)
        
    def forward(self, card_indices):
        # card_indices shape: (batch_size, sequence_length)
        card_embeds = self.card_embedding(card_indices)

        # Clamp indices to valid range for known cards (0-51).
        # Index 52 (unknown/hidden) would produce out-of-bounds suit index,
        # so clamp to 51 for suit/rank decomposition; the card_embedding
        # already handles index 52 correctly via its own embedding table.
        clamped = card_indices.clamp(max=51)
        suit_indices = torch.div(clamped, 13, rounding_mode='floor')
        rank_indices = clamped % 13

        # Get suit and rank embeddings
        suit_embeds = self.suit_embedding(suit_indices)
        rank_embeds = self.rank_embedding(rank_indices)

        # Combine all embeddings, ensure the size of card_embed (1, 6, 64) is same as suit_rank_emb (1, 6, 32)
        suit_rank_emb = torch.cat([suit_embeds, rank_embeds], dim=-1)
        return card_embeds + suit_rank_emb

class QTransformer(nn.Module):
    def __init__(self, embedding_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.card_embedding = CardEmbedding(embedding_dim)
        
        # Position encoding for sequence
        self.pos_encoding = nn.Parameter(torch.randn(1, 6, embedding_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Q-value prediction head
        self.q_head = nn.Sequential(
            nn.Linear(embedding_dim * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, player_cards, discard_top):
        # player_cards shape: (batch_size, 6)
        # discard_top shape: (batch_size, 1)
        
        # Get embeddings for player cards and discard pile top card
        player_embeds = self.card_embedding(player_cards)
        discard_embeds = self.card_embedding(discard_top)
        
        # Add position encoding to player cards
        player_embeds = player_embeds + self.pos_encoding
        
        # Combine player cards with discard top card
        sequence = torch.cat([player_embeds, discard_embeds], dim=1)
        
        # Pass through transformer
        transformed = self.transformer(sequence)
        
        # Flatten and predict Q-value
        flattened = transformed[:, :6].reshape(transformed.shape[0], -1)
        q_value = self.q_head(flattened)
        
        return q_value

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def train_episode(model, optimizer, replay_buffer, batch_size=32, gamma=0.99):
    if len(replay_buffer) < batch_size:
        return 0.0
        
    # Sample from replay buffer
    transitions = replay_buffer.sample(batch_size)
    batch = list(zip(*transitions))
    
    # Convert to tensors
    state_batch = torch.tensor(batch[0], dtype=torch.long)
    action_batch = torch.tensor(batch[1], dtype=torch.long)
    reward_batch = torch.tensor(batch[2], dtype=torch.float)
    next_state_batch = torch.tensor(batch[3], dtype=torch.long)
    done_batch = torch.tensor(batch[4], dtype=torch.float)
    
    # Calculate Q-values
    current_q_values = model(state_batch[:, :6], state_batch[:, 6:])
    next_q_values = model(next_state_batch[:, :6], next_state_batch[:, 6:])
    
    # Calculate target Q-values using Bellman equation
    target_q_values = reward_batch + gamma * (1 - done_batch) * next_q_values.detach()
    
    # Calculate loss
    loss = nn.MSELoss()(current_q_values, target_q_values)
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate_sequence(cards):
    """
    Calculate the value of a 6-card sequence.
    Implement your own reward function here based on game rules.
    """
    # Example: Simple sum of card values
    return sum(min((card % 13) + 1, 10) for card in cards)