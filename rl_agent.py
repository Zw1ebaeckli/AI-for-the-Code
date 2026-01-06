# Author: Jasmin Furler
"""
Reinforcement Learning Agent for CODE card game.
Uses Q-learning with function approximation.
"""

import random
import pickle
from typing import Tuple, Dict, List
from collections import defaultdict
from code_agent_v1 import (
    GameState, Card, setup_game, legal_moves, apply_move, 
    end_if_win, is_win, next_player_index, agent_decide_move,
    count_code_matches, simulate_move
)

# =============================================================================
# REWARD SYSTEM
# =============================================================================
REWARDS = {
    "win": 100,              # Won the game
    "lose": -100,            # Lost the game
    "timeout": -50,          # Game timed out
    "progress_gain": 10,     # Gained a code match
    "progress_loss": -15,    # Lost a code match
    "hand_reduce": 2,        # Reduced hand size (when progress >= 4)
    "hand_increase": -1,     # Hand size increased
    "play_action": 3,        # Played an action card
    "protect_code": 5,       # Kept a code-matching card
    "play_duplicate": 4,     # Played duplicate code card (smart)
    "forced_draw": -2,       # Had to draw (no good plays)
}


def get_state_features(game: GameState, player_idx: int) -> Tuple:
    """
    Extract features from game state for Q-learning.
    Returns a hashable tuple representing the state.
    """
    player = game.players[player_idx]
    hand = player.hand
    code_digits = set(player.code.target)
    
    # Feature 1: Hand size (bucketed)
    hand_size = len(hand)
    hand_bucket = min(hand_size // 2, 5)  # 0-1, 2-3, 4-5, 6-7, 8-9, 10+
    
    # Feature 2: Code progress (0-4)
    progress = count_code_matches(hand, player.code)
    
    # Feature 3: Number of duplicate code cards
    digit_counts = {}
    for c in hand:
        if c.kind == "NUMBER" and c.value in code_digits:
            digit_counts[c.value] = digit_counts.get(c.value, 0) + 1
    duplicates = sum(max(0, v - 1) for v in digit_counts.values())
    dup_bucket = min(duplicates, 3)
    
    # Feature 4: Action cards in hand (bucketed)
    action_count = sum(1 for c in hand if c.kind == "ACTION")
    action_bucket = min(action_count, 3)
    
    # Feature 5: Top discard value (for matching)
    top = game.number_discard[-1] if game.number_discard else None
    top_value = top.value if top else -1
    
    # Feature 6: Can play (has matching cards)
    moves = legal_moves(game)
    can_play = int(any(m[0] in ("PlayNumber", "PlaySum", "PlayAction") for m in moves))
    
    # Feature 7: Pending PLUS2
    pending = min(game.pending_plus2 // 2, 2)
    
    return (hand_bucket, progress, dup_bucket, action_bucket, top_value, can_play, pending)


def get_action_features(move_type: str, payload: Tuple, game: GameState, player_idx: int) -> Tuple:
    """
    Extract features from an action for Q(s,a) lookup.
    """
    player = game.players[player_idx]
    code_digits = set(player.code.target)
    
    # Action type encoding
    action_types = {"PlayNumber": 0, "PlaySum": 1, "PlayAction": 2, "PlayAny": 3, "Draw": 4}
    action_id = action_types.get(move_type, 5)
    
    # Is it playing a code card?
    plays_code = 0
    if move_type in ("PlayNumber", "PlayAny") and payload:
        if payload[0].kind == "NUMBER" and payload[0].value in code_digits:
            plays_code = 1
    elif move_type == "PlaySum" and payload:
        if (payload[0].value in code_digits) or (payload[1].value in code_digits):
            plays_code = 1
    
    # Action card type (if applicable)
    action_subtype = 0
    if move_type == "PlayAction" and payload:
        subtypes = {"PLUS2": 1, "AUSSETZEN": 2, "GESCHENK": 3, "TAUSCH": 4, "RICHTUNGSWECHSEL": 5, "RESET": 6}
        action_subtype = subtypes.get(payload[0].action, 0)
    
    return (action_id, plays_code, action_subtype)


class QLearningAgent:
    """
    Q-learning agent with state-action value function.
    """
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.2):
        """
        Args:
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate (for epsilon-greedy)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: Dict[Tuple, float] = defaultdict(float)
        self.training = True
        
    def get_q_value(self, state: Tuple, action: Tuple) -> float:
        """Get Q-value for state-action pair."""
        return self.q_table[(state, action)]
    
    def choose_action(self, game: GameState) -> Tuple[str, Tuple]:
        """
        Choose action using epsilon-greedy policy.
        """
        idx = game.active_idx
        moves = legal_moves(game)
        
        if not moves:
            return ("Pass", ())
        
        state = get_state_features(game, idx)
        
        # Epsilon-greedy exploration
        if self.training and random.random() < self.epsilon:
            return random.choice(moves)
        
        # Greedy: choose action with highest Q-value
        best_move = moves[0]
        best_q = float('-inf')
        
        for move_type, payload in moves:
            action = get_action_features(move_type, payload, game, idx)
            q = self.get_q_value(state, action)
            if q > best_q:
                best_q = q
                best_move = (move_type, payload)
        
        return best_move
    
    def update(self, state: Tuple, action: Tuple, reward: float, next_state: Tuple, 
               next_moves: List[Tuple[str, Tuple]], game: GameState, done: bool):
        """
        Q-learning update rule:
        Q(s,a) <- Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        """
        if not self.training:
            return
            
        current_q = self.get_q_value(state, action)
        
        if done:
            target = reward
        else:
            # Find max Q for next state
            max_next_q = float('-inf')
            idx = game.active_idx
            for move_type, payload in next_moves:
                next_action = get_action_features(move_type, payload, game, idx)
                max_next_q = max(max_next_q, self.get_q_value(next_state, next_action))
            if max_next_q == float('-inf'):
                max_next_q = 0
            target = reward + self.gamma * max_next_q
        
        # Update Q-value
        self.q_table[(state, action)] = current_q + self.alpha * (target - current_q)
    
    def save(self, filepath: str):
        """Save Q-table to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Saved {len(self.q_table)} Q-values to {filepath}")
    
    def load(self, filepath: str):
        """Load Q-table from file."""
        try:
            with open(filepath, 'rb') as f:
                self.q_table = defaultdict(float, pickle.load(f))
            print(f"Loaded {len(self.q_table)} Q-values from {filepath}")
        except FileNotFoundError:
            print(f"No saved model found at {filepath}, starting fresh.")


def compute_reward(game_before: GameState, game_after: GameState, move: Tuple[str, Tuple],
                   player_idx: int, winner: int = None) -> float:
    """
    Compute reward for a transition.
    """
    reward = 0.0
    
    # Terminal rewards
    if winner is not None:
        if winner == player_idx:
            return REWARDS["win"]
        else:
            return REWARDS["lose"]
    
    player_before = game_before.players[player_idx]
    player_after = game_after.players[player_idx]
    
    # Progress change
    progress_before = count_code_matches(player_before.hand, player_before.code)
    progress_after = count_code_matches(player_after.hand, player_after.code)
    
    if progress_after > progress_before:
        reward += REWARDS["progress_gain"] * (progress_after - progress_before)
    elif progress_after < progress_before:
        reward += REWARDS["progress_loss"] * (progress_before - progress_after)
    
    # Hand size change (reward reduction when we have enough progress)
    hand_before = len(player_before.hand)
    hand_after = len(player_after.hand)
    
    if progress_after >= 4:
        if hand_after < hand_before:
            reward += REWARDS["hand_reduce"] * (hand_before - hand_after)
    
    # Action card bonus
    move_type, payload = move
    if move_type == "PlayAction":
        reward += REWARDS["play_action"]
    
    # Draw penalty (when we had options)
    if move_type == "Draw":
        moves = legal_moves(game_before)
        if any(m[0] in ("PlayNumber", "PlaySum", "PlayAction") for m in moves):
            reward += REWARDS["forced_draw"]
    
    return reward


def resolve_plus2(game: GameState) -> GameState:
    """Resolve pending PLUS2 before a player's turn."""
    if game.pending_plus2 > 0:
        player = game.players[game.active_idx]
        if not any(c.kind == "ACTION" and c.action == "PLUS2" for c in player.hand):
            for _ in range(game.pending_plus2):
                if game.draw_pile:
                    player.hand.append(game.draw_pile.pop())
            game.pending_plus2 = 0
    return game


def train_agent(agent: QLearningAgent, opponent, n_episodes: int = 1000, 
                max_steps: int = 300, verbose: bool = True):
    """
    Train the RL agent by playing games.
    """
    agent.training = True
    wins = losses = timeouts = 0
    
    for episode in range(n_episodes):
        game = setup_game(num_players=2)
        
        for step in range(max_steps):
            game = resolve_plus2(game)
            idx = game.active_idx
            
            # Get state before move
            state = get_state_features(game, 0)  # Always track from P0's perspective
            
            # Choose action
            if idx == 0:
                move = agent.choose_action(game)
            else:
                move = opponent(game)
            
            move_type, payload = move
            action = get_action_features(move_type, payload, game, idx)
            
            # Apply move
            game_before = game
            game = apply_move(game, move_type, payload)
            
            if move_type != "PlayAction" or payload[0].action != "AUSSETZEN":
                game.active_idx = next_player_index(game, steps=1)
            
            # Check for win
            game, winner = end_if_win(game)
            
            # Compute reward and update (only for agent's moves)
            if idx == 0:
                reward = compute_reward(game_before, game, move, 0, winner)
                next_state = get_state_features(game, 0)
                next_moves = legal_moves(game) if winner is None else []
                agent.update(state, action, reward, next_state, next_moves, game, winner is not None)
            
            if winner is not None:
                if winner == 0:
                    wins += 1
                else:
                    losses += 1
                break
        else:
            timeouts += 1
            # Timeout penalty
            if idx == 0:
                agent.update(state, action, REWARDS["timeout"], state, [], game, True)
        
        # Decay epsilon
        if episode % 100 == 0 and agent.epsilon > 0.05:
            agent.epsilon *= 0.95
        
        # Progress report
        if verbose and (episode + 1) % 100 == 0:
            total = wins + losses + timeouts
            print(f"Episode {episode+1:4d} | Win: {wins/total:5.1%} | Loss: {losses/total:5.1%} | "
                  f"Timeout: {timeouts/total:5.1%} | Q-size: {len(agent.q_table)} | ε: {agent.epsilon:.3f}")
            wins = losses = timeouts = 0
    
    return agent


def evaluate_agent(agent: QLearningAgent, opponent, n_games: int = 500) -> Dict:
    """
    Evaluate agent performance (no learning).
    """
    agent.training = False
    wins = losses = timeouts = total_steps = 0
    
    for _ in range(n_games):
        game = setup_game(num_players=2)
        winner = None
        
        for step in range(300):
            game = resolve_plus2(game)
            idx = game.active_idx
            
            move = agent.choose_action(game) if idx == 0 else opponent(game)
            game = apply_move(game, move[0], move[1])
            
            if move[0] != "PlayAction" or move[1][0].action != "AUSSETZEN":
                game.active_idx = next_player_index(game, steps=1)
            
            game, winner = end_if_win(game)
            if winner is not None:
                break
        
        total_steps += step
        if winner == 0: wins += 1
        elif winner == 1: losses += 1
        else: timeouts += 1
    
    agent.training = True
    return {
        "wins": wins, "losses": losses, "timeouts": timeouts,
        "win_rate": wins / n_games, "avg_steps": total_steps / n_games
    }


def print_reward_table():
    """Print the reward system table."""
    print("\n" + "=" * 50)
    print("REWARD SYSTEM")
    print("=" * 50)
    print(f"{'Event':<25} | {'Reward':>10}")
    print("-" * 50)
    for event, reward in REWARDS.items():
        sign = "+" if reward > 0 else ""
        print(f"{event:<25} | {sign}{reward:>9}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    print_reward_table()
    
    # Create and train agent
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.3)
    
    print("Training RL agent against rule-bot...")
    print("-" * 75)
    train_agent(agent, agent_decide_move, n_episodes=500, verbose=True)
    
    # Evaluate
    print("\n" + "=" * 75)
    print("Evaluation (500 games, no exploration)")
    print("=" * 75)
    
    results = evaluate_agent(agent, agent_decide_move, n_games=500)
    print(f"RL Agent vs Rule-bot | Win: {results['win_rate']:5.1%} | "
          f"Loss: {results['losses']/500:5.1%} | Timeout: {results['timeouts']/500:5.1%} | "
          f"Steps: {results['avg_steps']:.0f}")
    
    # Save model
    agent.save("rl_agent_model.pkl")
