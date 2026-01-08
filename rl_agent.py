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
    "progress_gain": 15,     # Gained a code match (increased from 10)
    "progress_loss": -20,    # Lost a code match (more severe)
    "hand_reduce": 1,        # Reduced hand size (lowered from 2 - less aggressive)
    "hand_reduce_smart": 3,  # Reduced hand by playing duplicate/non-code card
    "hand_increase": -1,     # Hand size increased
    "play_action": 2,        # Played an action card (slightly reduced)
    "protect_code": 8,       # Successfully kept code-matching card (increased)
    "play_duplicate": 5,     # Played duplicate code card - smart move
    "play_non_code": 2,      # Played non-code card (better than code card)
    "forced_draw": -3,       # Had to draw when plays available (increased penalty)
    "endgame_hand_reduce": 5,# Reduced hand when close to winning (progress = 4)
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
    
    # Feature 5: Non-code number cards (safe to play)
    non_code_count = sum(1 for c in hand if c.kind == "NUMBER" and c.value not in code_digits)
    non_code_bucket = min(non_code_count // 2, 3)
    
    # Feature 6: Top discard value (for matching)
    top = game.number_discard[-1] if game.number_discard else None
    top_value = top.value if top else -1
    
    # Feature 7: Can play (has matching cards)
    moves = legal_moves(game)
    can_play = int(any(m[0] in ("PlayNumber", "PlaySum", "PlayAction") for m in moves))
    
    # Feature 8: Pending PLUS2
    pending = min(game.pending_plus2 // 2, 2)
    
    return (hand_bucket, progress, dup_bucket, action_bucket, non_code_bucket, top_value, can_play, pending)


def get_action_features(move_type: str, payload: Tuple, game: GameState, player_idx: int) -> Tuple:
    """
    Extract features from an action for Q(s,a) lookup.
    """
    player = game.players[player_idx]
    code_digits = set(player.code.target)
    
    # Action type encoding
    action_types = {"PlayNumber": 0, "PlaySum": 1, "PlayAction": 2, "PlayAny": 3, "Draw": 4}
    action_id = action_types.get(move_type, 5)
    
    # Card value analysis for the move
    plays_code = 0
    is_duplicate = 0
    
    if move_type in ("PlayNumber", "PlayAny") and payload:
        card = payload[0]
        if card.kind == "NUMBER":
            # Is it a code card?
            if card.value in code_digits:
                plays_code = 1
                # Is it a duplicate?
                same_value_count = sum(1 for c in player.hand if c.kind == "NUMBER" and c.value == card.value)
                if same_value_count > 1:
                    is_duplicate = 1
    
    elif move_type == "PlaySum" and payload:
        card_a, card_b = payload[0], payload[1]
        # Count code cards being played
        code_count = 0
        if card_a.value in code_digits:
            code_count += 1
        if card_b.value in code_digits:
            code_count += 1
        
        plays_code = code_count  # 0, 1, or 2
        
        # Check if either is a duplicate
        count_a = sum(1 for c in player.hand if c.kind == "NUMBER" and c.value == card_a.value)
        count_b = sum(1 for c in player.hand if c.kind == "NUMBER" and c.value == card_b.value)
        if count_a > 1 or count_b > 1:
            is_duplicate = 1
    
    # Action card type (if applicable)
    action_subtype = 0
    if move_type == "PlayAction" and payload:
        subtypes = {"PLUS2": 1, "AUSSETZEN": 2, "GESCHENK": 3, "TAUSCH": 4, "RICHTUNGSWECHSEL": 5, "RESET": 6}
        action_subtype = subtypes.get(payload[0].action, 0)
    
    return (action_id, plays_code, is_duplicate, action_subtype)


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
    code_digits = set(player_before.code.target)
    
    # Progress change
    progress_before = count_code_matches(player_before.hand, player_before.code)
    progress_after = count_code_matches(player_after.hand, player_after.code)
    
    if progress_after > progress_before:
        reward += REWARDS["progress_gain"] * (progress_after - progress_before)
    elif progress_after < progress_before:
        reward += REWARDS["progress_loss"] * (progress_before - progress_after)
    
    # Hand size change - be smarter about what we're playing
    hand_before = len(player_before.hand)
    hand_after = len(player_after.hand)
    move_type, payload = move
    
    # Check if we played a card intelligently
    played_card_info = None
    if move_type in ("PlayNumber", "PlayAny") and payload:
        card = payload[0]
        if card.kind == "NUMBER":
            is_code_card = card.value in code_digits
            # Count duplicates in hand
            same_value_count = sum(1 for c in player_before.hand if c.kind == "NUMBER" and c.value == card.value)
            is_duplicate = same_value_count > 1
            played_card_info = (is_code_card, is_duplicate)
    elif move_type == "PlaySum" and payload:
        # For sum plays, check if we played duplicates or non-code cards
        card_a, card_b = payload[0], payload[1]
        is_code_a = card_a.value in code_digits
        is_code_b = card_b.value in code_digits
        
        # Count how many of each value we have
        count_a = sum(1 for c in player_before.hand if c.kind == "NUMBER" and c.value == card_a.value)
        count_b = sum(1 for c in player_before.hand if c.kind == "NUMBER" and c.value == card_b.value)
        
        # Best case: both are duplicates or non-code
        if (count_a > 1 and count_b > 1) or (not is_code_a and not is_code_b):
            played_card_info = (False, True)  # Treat as smart play
        elif count_a > 1 or count_b > 1:
            played_card_info = (is_code_a and is_code_b, True)  # Mixed
        else:
            played_card_info = (is_code_a or is_code_b, False)
    
    # Reward hand reduction based on what was played
    if hand_after < hand_before:
        reduction = hand_before - hand_after
        
        if progress_after == 4:
            # Endgame: aggressively reduce hand
            reward += REWARDS["endgame_hand_reduce"] * reduction
        elif played_card_info:
            is_code_card, is_duplicate = played_card_info
            
            if is_duplicate and is_code_card:
                # Played a duplicate code card - smart!
                reward += REWARDS["play_duplicate"] * reduction
            elif not is_code_card:
                # Played a non-code card - good
                reward += REWARDS["play_non_code"] * reduction
            elif is_code_card and not is_duplicate:
                # Played a valuable unique code card - only small reward
                reward += REWARDS["hand_reduce"] * reduction * 0.5
            else:
                reward += REWARDS["hand_reduce_smart"] * reduction
        else:
            # Action card or other play
            reward += REWARDS["hand_reduce"] * reduction
    
    # Protect code bonus: reward for keeping code cards when we could have played them
    if progress_before < 4 and progress_after < 4:
        # Count code-matching cards kept
        code_cards_kept = sum(1 for c in player_after.hand if c.kind == "NUMBER" and c.value in code_digits)
        if code_cards_kept >= progress_after and move_type in ("PlayNumber", "PlaySum", "PlayAny"):
            # Successfully kept code cards while playing
            reward += REWARDS["protect_code"] * 0.1  # Small bonus for conservation
    
    # Action card bonus
    if move_type == "PlayAction":
        reward += REWARDS["play_action"]
    
    # Draw penalty (when we had play options)
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
