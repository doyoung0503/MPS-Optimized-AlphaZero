"""
Monte Carlo Tree Search with Batch Processing
"""
import torch
import numpy as np
from chess_game import Chess

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 0.0
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0
    
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct, parent_visits):
        prior_score = c_puct * self.prior * np.sqrt(parent_visits) / (1 + self.visit_count)
        if self.visit_count > 0:
            value_score = self.value()
        else:
            value_score = 0.0
        return value_score + prior_score

class MCTS:
    def __init__(self, model, num_simulations=100, c_puct=1.5, device='cpu'):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        self.model.eval()
    
    def search_batch(self, states):
        """
        Batch MCTS search
        Args:
            states: List of Chess game states
        Returns:
            policies: List of policy distributions (4352,)
        """
        batch_size = len(states)
        roots = [MCTSNode(state.copy()) for state in states]
        
        # Initial expansion
        self._expand_batch(roots)
        
        # Simulations
        for _ in range(self.num_simulations):
            nodes_to_expand = []
            paths = []
            
            # Selection
            for root in roots:
                node, path = self._select(root)
                if not node.is_expanded and not node.is_terminal:
                    nodes_to_expand.append(node)
                    paths.append(path)
            
            # Expansion & Evaluation
            if nodes_to_expand:
                self._expand_batch(nodes_to_expand)
            
            # Backpropagation
            for path in paths:
                if path:
                    leaf = path[-1]
                    value = leaf.value() if leaf.visit_count > 0 else 0.0
                    self._backprop(path, value)
        
        # Extract policies
        policies = []
        for root in roots:
            policy = np.zeros(4352, dtype=np.float32)
            total_visits = sum(child.visit_count for child in root.children.values())
            if total_visits > 0:
                for action, child in root.children.items():
                    policy[action] = child.visit_count / total_visits
            policies.append(policy)
        
        return policies
    
    def _select(self, root):
        """Select path from root to leaf"""
        path = []
        node = root
        
        while node.is_expanded and not node.is_terminal:
            path.append(node)
            
            # Select best child
            best_score = -float('inf')
            best_action = None
            best_child = None
            
            for action, child in node.children.items():
                score = child.ucb_score(self.c_puct, node.visit_count)
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_child = child
            
            if best_child is None:
                break
            node = best_child
        
        path.append(node)
        return node, path
    
    def _expand_batch(self, nodes):
        """Expand a batch of nodes"""
        if not nodes:
            return
        
        # Collect states
        states_list = []
        for node in nodes:
            canonical = node.state.get_canonical()
            states_list.append(canonical.encode())
        
        # Neural network inference
        with torch.no_grad():
            states_tensor = torch.FloatTensor(np.array(states_list)).to(self.device)
            log_pis, values = self.model(states_tensor)
            policies = torch.exp(log_pis).cpu().numpy()
            values = values.cpu().numpy().flatten()
        
        # Expand each node
        for i, node in enumerate(nodes):
            policy = policies[i]
            value = float(values[i])
            
            # Check terminal
            if node.state.is_game_over():
                node.is_terminal = True
                winner = node.state.get_winner()
                # Value from current player's perspective
                node.terminal_value = float(winner) * node.state.turn
                continue
            
            # Get legal moves
            legal_moves = node.state.get_legal_moves()
            
            # If turn is Black, flip policy back to real perspective
            if node.state.turn == -1:
                policy = self._mirror_policy(policy)
            
            # Mask & normalize policy
            legal_probs = []
            legal_actions = []
            for move in legal_moves:
                from_sq, to_sq = move
                action = from_sq * 64 + to_sq
                if action < 4352:
                    legal_probs.append(policy[action])
                    legal_actions.append(action)
            
            total = sum(legal_probs) if legal_probs else 0.0
            if total == 0:
                legal_probs = [1.0 / len(legal_actions)] * len(legal_actions) if legal_actions else []
            else:
                legal_probs = [p / total for p in legal_probs]
            
            # Create children
            for action, prob in zip(legal_actions, legal_probs):
                from_sq, to_sq = action // 64, action % 64
                child_state = node.state.copy()
                child_state.push((from_sq, to_sq))
                child = MCTSNode(child_state, parent=node)
                child.prior = prob
                node.children[action] = child
            
            node.is_expanded = True
            # Initial value estimate
            node.visit_count = 1
            node.value_sum = value
    
    def _mirror_policy(self, policy):
        """Mirror policy for Black player"""
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
    
    def _backprop(self, path, value):
        """Backpropagate value up the tree"""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip perspective
    
    def select_action(self, policy, temperature=1.0):
        """Select action from policy distribution"""
        if temperature < 0.1:
            return np.argmax(policy)
        else:
            policy = policy ** (1.0 / temperature)
            policy /= policy.sum()
            return np.random.choice(len(policy), p=policy)
