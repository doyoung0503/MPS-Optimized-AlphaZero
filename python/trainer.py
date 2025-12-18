"""
AlphaZero Training Loop
"""
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import time

from chess_game import Chess
from network import AlphaZeroNet
from mcts import MCTS

class Trainer:
    def __init__(self, model, device='cpu', lr=0.001):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.buffer = deque(maxlen=30000)
        self.batch_size = 64
    
    def self_play(self, num_games, num_simulations):
        """Run self-play games"""
        print(f"[Self-Play] Playing {num_games} games...")
        self.model.eval()
        
        mcts = MCTS(self.model, num_simulations, device=self.device)
        
        # Initialize games
        games = [Chess() for _ in range(num_games)]
        histories = [[] for _ in range(num_games)]
        finished = [False] * num_games
        
        moves_count = 0
        while not all(finished):
            # Active games
            active_states = []
            active_indices = []
            for i, game in enumerate(games):
                if not finished[i]:
                    active_states.append(game)
                    active_indices.append(i)
            
            if not active_states:
                break
            
            # MCTS search
            policies = mcts.search_batch(active_states)
            
            # Apply moves
            for k, idx in enumerate(active_indices):
                game = games[idx]
                policy = policies[k]
                
                # Normalize policy (sanity check)
                policy_sum = policy.sum()
                if policy_sum < 1e-8:
                    # Shouldn't happen, but fallback
                    legal_moves = game.get_legal_moves()
                    policy = np.zeros(4352, dtype=np.float32)
                    for move in legal_moves:
                        from_sq, to_sq = move
                        action = from_sq * 64 + to_sq
                        if action < 4352:
                            policy[action] = 1.0
                    policy /= policy.sum() if policy.sum() > 0 else 1.0
                else:
                    policy = policy / policy_sum
                
                # Store history (state, policy, player)
                histories[idx].append({
                    'state': game.copy(),
                    'policy': policy,
                    'player': game.turn
                })
                
                # Select action (temperature annealing)
                temperature = 1.0 if moves_count < 30 else 0.1
                action = mcts.select_action(policy, temperature)
                
                # Make move
                from_sq, to_sq = action // 64, action % 64
                game.push((from_sq, to_sq))
                
                # Check game over
                if game.is_game_over() or len(histories[idx]) > 200:
                    finished[idx] = True
                    
                    # Get outcome
                    winner = game.get_winner()
                    if len(histories[idx]) > 200:
                        winner = 0
                    
                    # Store to buffer with proper canonical forms
                    for step in histories[idx]:
                        value = float(winner * step['player'])
                        
                        # Convert policy to canonical if player was Black
                        train_policy = step['policy'].copy()
                        if step['player'] == -1:
                            train_policy = self._mirror_policy(train_policy)
                        
                        self.buffer.append({
                            'state': step['state'],
                            'policy': train_policy,
                            'value': value
                        })
            
            moves_count += 1
            
            # Progress
            if moves_count % 10 == 0:
                active = sum(1 for f in finished if not f)
                print(f"\r  Active: {active}, Finished: {sum(finished)}, Moves: {moves_count}", end='', flush=True)
        
        print(f"\n  Self-play complete. Buffer size: {len(self.buffer)}")
    
    def _mirror_policy(self, policy):
        """Mirror policy for canonical form"""
        mirrored = np.zeros_like(policy)
        for a in range(4352):
            from_sq = a // 64
            to_sq = a % 64
            from_r, from_c = from_sq // 8, from_sq % 8
            to_r, to_c = to_sq // 8, to_sq % 8
            # Mirror rows
            m_from_sq = (7 - from_r) * 8 + from_c
            m_to_sq = (7 - to_r) * 8 + to_c
            m_action = m_from_sq * 64 + m_to_sq
            if m_action < 4352:
                mirrored[m_action] = policy[a]
        return mirrored
    
    def training_step(self, num_batches):
        """Train on replay buffer"""
        if len(self.buffer) < self.batch_size:
            return
        
        print(f"[Training] Running {num_batches} batches...")
        self.model.train()
        
        total_loss = 0.0
        for b in range(num_batches):
            # Sample batch
            indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
            
            # Prepare tensors
            states = []
            target_pis = []
            target_vs = []
            
            for sample in batch:
                canonical = sample['state'].get_canonical()
                states.append(canonical.encode())
                target_pis.append(sample['policy'])
                target_vs.append([sample['value']])
            
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            pis_tensor = torch.FloatTensor(np.array(target_pis)).to(self.device)
            vs_tensor = torch.FloatTensor(np.array(target_vs)).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            log_pis, values = self.model(states_tensor)
            
            # Loss calculation
            # Policy loss: cross-entropy (target * log_pred)
            p_loss = -torch.sum(pis_tensor * log_pis) / self.batch_size
            
            # Value loss: MSE
            v_loss = nn.MSELoss()(values, vs_tensor)
            
            # Total loss
            loss = p_loss + v_loss
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if b == 0:
                print(f"  [Debug] p_loss: {p_loss.item():.4f}, v_loss: {v_loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        print(f"  Avg Loss: {avg_loss:.4f}")
        return avg_loss
    
    def train(self, iterations, games_per_iter, max_sims, train_batches):
        """Main training loop"""
        print("=" * 50)
        print("  Python AlphaZero Training")
        print("=" * 50)
        print(f"Iterations: {iterations}")
        print(f"Games/Iter: {games_per_iter}")
        print(f"Simulations: {max_sims}")
        print()
        
        for iteration in range(1, iterations + 1):
            print(f"----- Iteration {iteration}/{iterations} -----")
            
            start = time.time()
            
            # Self-play
            self.self_play(games_per_iter, max_sims)
            
            # Training
            loss = self.training_step(train_batches)
            
            elapsed = time.time() - start
            print(f"  Time: {elapsed:.1f}s")
            print()
