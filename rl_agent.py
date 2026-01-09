# Author: Jasmin Furler
"""
Reinforcement Learning Agent for CODE card game.
Uses Double Q-learning with function approximation.

DOUBLE Q-LEARNING:
Standard Q-learning suffers from overestimation bias because it uses the same
values to select and evaluate actions (max operator). This leads to overly
optimistic Q-values and can hurt performance.

Double Q-Learning fixes this by:
1. Maintaining TWO Q-tables (Q_A and Q_B)
2. For each update, randomly choosing which table to update
3. Using one table to SELECT the best action, the other to EVALUATE it

Example:
- Update Q_A: use Q_A to find argmax_a Q_A(s',a), then use Q_B(s', a*) as value
- Update Q_B: use Q_B to find argmax_a Q_B(s',a), then use Q_A(s', a*) as value

This decoupling reduces overestimation and leads to more stable, accurate learning.

ADDITIONAL FEATURES:
- Optimistic initialization (encourages exploration)
- Adaptive learning rate (decreases with visit count)
- Experience replay (learns from stored transitions)
- Curriculum learning (easy -> hard opponents)
- Phase-aware rewards (endgame urgency)
- Enhanced state features (10 strategic indicators)
"""

import random
import pickle
from typing import Tuple, Dict, List
from collections import defaultdict, deque

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
from code_agent_v1 import (
    GameState, Card, setup_game, legal_moves, apply_move, 
    end_if_win, is_win, next_player_index, agent_decide_move,
    count_code_matches, simulate_move
)

# =============================================================================
# REWARD SYSTEM (Simplified for clarity and stability)
# =============================================================================
REWARDS = {
    "win": 200,              # Terminal: Won the game (strong signal)
    "lose": -100,            # Terminal: Lost the game
    "timeout": -50,          # Terminal: Game timed out
    "progress_gain": 40,     # Gained a code match (PRIMARY objective, boosted)
    "progress_loss": -40,    # Lost a code match (symmetric penalty)
    "milestone_3": 20,       # Reached 3 code matches (close to winning)
    "milestone_4": 30,       # Reached 4 code matches (ready to win)
    "hand_reduce": 2,        # Reduced hand size (general progress)
    "hand_reduce_smart": 12, # Played duplicate or non-code (strategic, boosted)
    "forced_draw": -6,       # Had to draw when plays existed (bad choice, harsher)
    "opp_hand_pressure": 5,  # Opponent hand grew (defensive good)
    "action_play_bonus": 1,  # Played action card (minor incentive)
}


def get_state_features(game: GameState, player_idx: int) -> Tuple:
    """
    Extract features from game state for Q-learning.
    Enhanced with strategic gameplay indicators.
    """
    player = game.players[player_idx]
    hand = player.hand
    code_digits = set(player.code.target)
    opp_idx = 1 if player_idx == 0 else 0
    opponent = game.players[opp_idx]
    
    # Core progress metrics
    progress = count_code_matches(hand, player.code)
    hand_size = len(hand)
    opp_hand_size = len(opponent.hand)
    
    # Strategic indicators
    code_cards_in_hand = sum(1 for c in hand if c.kind == "NUMBER" and c.value in code_digits)
    duplicates = sum(1 for v in code_digits 
                    if sum(1 for c in hand if c.kind == "NUMBER" and c.value == v) > 1)
    
    # Can we make progress? (have code cards we don't match yet)
    missing_codes = [d for d in code_digits 
                     if not any(c.kind == "NUMBER" and c.value == d for c in hand)]
    can_improve = int(len(missing_codes) > 0 and code_cards_in_hand > progress)
    
    # Playability: do we have moves?
    moves = legal_moves(game)
    has_play = int(any(m[0] in ("PlayNumber", "PlaySum", "PlayAction") for m in moves))
    can_play_code = int(any(m[0] in ("PlayNumber", "PlaySum") and 
                           any(c.kind == "NUMBER" and c.value in code_digits 
                               for c in ([m[1][0]] if m[0] == "PlayNumber" else m[1][:2]))
                           for m in moves if m[1]))
    
    # Game phase indicator (0=early, 1=mid, 2=endgame)
    if progress >= 4 or opp_hand_size <= 2:
        phase = 2  # Endgame: must finish
    elif progress >= 2:
        phase = 1  # Mid-game: building
    else:
        phase = 0  # Early: exploring
    
    # Top discard
    top = game.number_discard[-1] if game.number_discard else None
    top_value = top.value if top else -1
    
    return (
        progress,                    # 0-4: Our code progress (PRIMARY)
        min(hand_size // 2, 5),      # 0-5: Our hand size bucketed
        min(opp_hand_size // 2, 5),  # 0-5: Opponent hand size
        min(code_cards_in_hand, 4),  # 0-4: Code cards we have
        min(duplicates, 3),          # 0-3: Duplicate code cards (safe to play)
        can_improve,                 # 0-1: Can we still gain progress?
        has_play,                    # 0-1: Can we play anything?
        can_play_code,               # 0-1: Would playing lose progress?
        phase,                       # 0-2: Game phase
        min(game.pending_plus2, 4),  # 0-4: PLUS2 pressure
    )


def get_action_features(move_type: str, payload: Tuple, game: GameState, player_idx: int) -> Tuple:
    """
    Extract features from an action for Q(s,a) lookup.
    Simplified and more strategic.
    """
    player = game.players[player_idx]
    code_digits = set(player.code.target)
    
    # Action type encoding (simplified)
    action_types = {"PlayNumber": 0, "PlaySum": 1, "PlayAction": 2, "Draw": 3}
    action_id = action_types.get(move_type, 3)
    
    # Strategic value: is this a smart play? (0=bad, 1=neutral, 2=smart)
    play_quality = 1  # Default: neutral
    
    if move_type in ("PlayNumber", "PlayAny") and payload:
        card = payload[0]
        if card.kind == "NUMBER":
            same_count = sum(1 for c in player.hand if c.kind == "NUMBER" and c.value == card.value)
            is_duplicate = same_count > 1
            is_code = card.value in code_digits
            
            # Smart: duplicate or non-code, Bad: unique code card
            if is_duplicate or not is_code:
                play_quality = 2
            elif is_code and not is_duplicate:
                play_quality = 0
    
    elif move_type == "PlaySum" and payload:
        card_a, card_b = payload[0], payload[1]
        count_a = sum(1 for c in player.hand if c.kind == "NUMBER" and c.value == card_a.value)
        count_b = sum(1 for c in player.hand if c.kind == "NUMBER" and c.value == card_b.value)
        
        # Smart if playing duplicates or non-code
        if (count_a > 1 or count_b > 1):
            play_quality = 2
        elif (card_a.value not in code_digits and card_b.value not in code_digits):
            play_quality = 2
    
    elif move_type == "PlayAction":
        play_quality = 1  # Actions are generally acceptable
    
    elif move_type == "Draw":
        # Drawing is bad if we have plays available
        moves = legal_moves(game)
        if any(m[0] in ("PlayNumber", "PlaySum", "PlayAction") for m in moves):
            play_quality = 0
    
    return (action_id, play_quality)


class QLearningAgent:
    """
    Double Q-learning agent with state-action value function.
    Uses two Q-tables to reduce overestimation bias.
    Enhanced with optimistic initialization, adaptive learning, and experience replay.
    """
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.2, 
                 optimistic_init: float = 5.0, replay_size: int = 10000):
        """
        Args:
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate (for epsilon-greedy)
            optimistic_init: Initial Q-value (encourages exploration)
            replay_size: Size of experience replay buffer
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimistic_init = optimistic_init
        
        # Double Q-Learning: maintain two Q-tables
        self.q_table_a: Dict[Tuple, float] = defaultdict(lambda: optimistic_init)
        self.q_table_b: Dict[Tuple, float] = defaultdict(lambda: optimistic_init)
        
        self.training = True
        self.visit_counts: Dict[Tuple, int] = defaultdict(int)
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=replay_size)
        self.replay_batch_size = 32
        
    def get_q_value(self, state: Tuple, action: Tuple, table: str = 'avg') -> float:
        """
        Get Q-value for state-action pair.
        
        Args:
            state: State tuple
            action: Action tuple
            table: Which table to use ('a', 'b', or 'avg' for average)
        """
        if table == 'a':
            return self.q_table_a[(state, action)]
        elif table == 'b':
            return self.q_table_b[(state, action)]
        else:  # Average of both tables (default for action selection)
            return (self.q_table_a[(state, action)] + self.q_table_b[(state, action)]) / 2
    
    def choose_action(self, game: GameState) -> Tuple[str, Tuple]:
        """
        Choose action using epsilon-greedy policy with strategic tie-breaking.
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
        best_moves = []  # Track ties
        
        for move_type, payload in moves:
            action = get_action_features(move_type, payload, game, idx)
            q = self.get_q_value(state, action)
            
            if q > best_q:
                best_q = q
                best_move = (move_type, payload)
                best_moves = [(move_type, payload)]
            elif q == best_q:
                best_moves.append((move_type, payload))
        
        # If multiple moves have same Q-value, prefer non-draw actions
        if len(best_moves) > 1:
            non_draw = [m for m in best_moves if m[0] != "Draw"]
            if non_draw:
                return random.choice(non_draw)
        
        return best_move
    
    def update(self, state: Tuple, action: Tuple, reward: float, next_state: Tuple, 
               next_moves: List[Tuple[str, Tuple]], game: GameState, done: bool):
        """
        Q-learning update with experience replay.
        Stores transition and learns from both current and replayed experiences.
        """
        if not self.training:
            return
        
        # Store experience in replay buffer
        self.replay_buffer.append((state, action, reward, next_state, next_moves, game, done))
        
        # Learn from current experience
        self._update_q_value(state, action, reward, next_state, next_moves, game, done)
        
        # Replay experiences if buffer has enough samples
        if len(self.replay_buffer) >= self.replay_batch_size:
            # Sample random batch from replay buffer
            batch = random.sample(self.replay_buffer, min(self.replay_batch_size, len(self.replay_buffer)))
            for s, a, r, ns, nm, g, d in batch:
                self._update_q_value(s, a, r, ns, nm, g, d)
    
    def _update_q_value(self, state: Tuple, action: Tuple, reward: float, next_state: Tuple,
                        next_moves: List[Tuple[str, Tuple]], game: GameState, done: bool):
        """
        Double Q-Learning update with adaptive learning rate.
        Randomly updates one of the two Q-tables, using the other to evaluate.
        This reduces overestimation bias.
        """
        # Track visits
        sa_pair = (state, action)
        self.visit_counts[sa_pair] += 1
        
        # Adaptive learning rate (decreases with visits for stability)
        visit_count = self.visit_counts[sa_pair]
        adaptive_alpha = self.alpha / (1 + 0.01 * visit_count)
        
        # Randomly choose which Q-table to update (Double Q-Learning)
        update_a = random.random() < 0.5
        
        if done:
            target = reward
        else:
            # Double Q-Learning: decouple action selection from evaluation
            idx = game.active_idx
            
            if update_a:
                # Update Q_A: use Q_A to select action, Q_B to evaluate
                best_action = None
                best_q_a = float('-inf')
                for move_type, payload in next_moves:
                    next_action = get_action_features(move_type, payload, game, idx)
                    q_a = self.q_table_a[(next_state, next_action)]
                    if q_a > best_q_a:
                        best_q_a = q_a
                        best_action = next_action
                
                # Evaluate best action using Q_B
                if best_action is not None:
                    max_next_q = self.q_table_b[(next_state, best_action)]
                else:
                    max_next_q = 0
            else:
                # Update Q_B: use Q_B to select action, Q_A to evaluate
                best_action = None
                best_q_b = float('-inf')
                for move_type, payload in next_moves:
                    next_action = get_action_features(move_type, payload, game, idx)
                    q_b = self.q_table_b[(next_state, next_action)]
                    if q_b > best_q_b:
                        best_q_b = q_b
                        best_action = next_action
                
                # Evaluate best action using Q_A
                if best_action is not None:
                    max_next_q = self.q_table_a[(next_state, best_action)]
                else:
                    max_next_q = 0
            
            target = reward + self.gamma * max_next_q
        
        # Update the chosen Q-table
        if update_a:
            current_q = self.q_table_a[(state, action)]
            self.q_table_a[(state, action)] = current_q + adaptive_alpha * (target - current_q)
        else:
            current_q = self.q_table_b[(state, action)]
            self.q_table_b[(state, action)] = current_q + adaptive_alpha * (target - current_q)
    
    def save(self, filepath: str):
        """Save both Q-tables to file."""
        save_data = {
            'q_table_a': dict(self.q_table_a),
            'q_table_b': dict(self.q_table_b),
            'visit_counts': dict(self.visit_counts)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Saved {len(self.q_table_a)} Q-values (table A) and {len(self.q_table_b)} Q-values (table B) to {filepath}")
    
    def load(self, filepath: str):
        """Load both Q-tables from file."""
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # Handle both old (single table) and new (double table) formats
            if isinstance(save_data, dict) and 'q_table_a' in save_data:
                # New format with double Q-tables
                self.q_table_a = defaultdict(lambda: self.optimistic_init, save_data['q_table_a'])
                self.q_table_b = defaultdict(lambda: self.optimistic_init, save_data['q_table_b'])
                if 'visit_counts' in save_data:
                    self.visit_counts = defaultdict(int, save_data['visit_counts'])
                print(f"Loaded {len(self.q_table_a)} Q-values (table A) and {len(self.q_table_b)} Q-values (table B) from {filepath}")
            else:
                # Old format with single table - duplicate to both tables
                self.q_table_a = defaultdict(lambda: self.optimistic_init, save_data)
                self.q_table_b = defaultdict(lambda: self.optimistic_init, save_data)
                print(f"Loaded {len(self.q_table_a)} Q-values from legacy format to both tables")
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
    opp_idx = 1 if player_idx == 0 else 0
    opp_before = game_before.players[opp_idx]
    opp_after = game_after.players[opp_idx]
    code_digits = set(player_before.code.target)
    
    # Determine game phase for adaptive rewards
    progress_before = count_code_matches(player_before.hand, player_before.code)
    progress_after = count_code_matches(player_after.hand, player_after.code)
    opp_hand_size = len(opp_after.hand)
    
    # Game phase: 0=early, 1=mid, 2=endgame
    if progress_after >= 4 or opp_hand_size <= 2:
        phase = 2  # Endgame: urgency to finish
    elif progress_after >= 2:
        phase = 1  # Mid-game
    else:
        phase = 0  # Early game
    
    # Progress change (PRIMARY reward signal with phase scaling)
    if progress_after > progress_before:
        progress_reward = REWARDS["progress_gain"] * (progress_after - progress_before)
        # Scale up in endgame (more urgent)
        if phase == 2:
            progress_reward *= 1.5
        reward += progress_reward
        
        # Milestone bonuses for getting close to winning
        if progress_after >= 4 and progress_before < 4:
            reward += REWARDS["milestone_4"]
        elif progress_after >= 3 and progress_before < 3:
            reward += REWARDS["milestone_3"]
    elif progress_after < progress_before:
        penalty = REWARDS["progress_loss"] * (progress_before - progress_after)
        # Harsher penalty in endgame
        if phase == 2:
            penalty *= 1.5
        reward += penalty
    
    # Hand size change
    hand_before = len(player_before.hand)
    hand_after = len(player_after.hand)
    opp_hand_before = len(opp_before.hand)
    opp_hand_after = len(opp_after.hand)
    move_type, payload = move
    
    # Reward smart hand reduction
    if hand_after < hand_before:
        reduction = hand_before - hand_after
        
        # Check if we played duplicates or non-code cards (smarter moves)
        if move_type in ("PlayNumber", "PlayAny") and payload:
            card = payload[0]
            if card.kind == "NUMBER":
                same_count = sum(1 for c in player_before.hand if c.kind == "NUMBER" and c.value == card.value)
                is_duplicate = same_count > 1
                is_code = card.value in code_digits
                
                if is_duplicate or not is_code:
                    reward += REWARDS["hand_reduce_smart"] * reduction
                else:
                    reward += REWARDS["hand_reduce"] * reduction
        elif move_type == "PlaySum" and payload:
            card_a, card_b = payload[0], payload[1]
            count_a = sum(1 for c in player_before.hand if c.kind == "NUMBER" and c.value == card_a.value)
            count_b = sum(1 for c in player_before.hand if c.kind == "NUMBER" and c.value == card_b.value)
            
            if (count_a > 1 or count_b > 1) or (card_a.value not in code_digits and card_b.value not in code_digits):
                reward += REWARDS["hand_reduce_smart"] * reduction
            else:
                reward += REWARDS["hand_reduce"] * reduction
        else:
            reward += REWARDS["hand_reduce"] * reduction
    
    # Opponent hand pressure (when opponent hand grows, that's good for us)
    if opp_hand_after > opp_hand_before:
        reward += REWARDS["opp_hand_pressure"] * (opp_hand_after - opp_hand_before)
    
    # Action card bonus
    if move_type == "PlayAction":
        reward += REWARDS["action_play_bonus"]
    
    # Draw penalty (when we had plays available but drew anyway)
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
                max_steps: int = 300, verbose: bool = True, history: List[Dict] = None,
                progress_every: int = 100):
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
        if (episode + 1) % progress_every == 0:
            total = wins + losses + timeouts
            win_rate = wins / total if total else 0.0
            loss_rate = losses / total if total else 0.0
            timeout_rate = timeouts / total if total else 0.0
            if history is not None:
                history.append({
                    "episode": episode + 1,
                    "win_rate": win_rate,
                    "loss_rate": loss_rate,
                    "timeout_rate": timeout_rate,
                    "epsilon": agent.epsilon,
                    "q_size": len(agent.q_table_a) + len(agent.q_table_b),
                })
            if verbose:
                print(f"Episode {episode+1:4d} | Win: {win_rate:5.1%} | Loss: {loss_rate:5.1%} | "
                      f"Timeout: {timeout_rate:5.1%} | Q-size: {len(agent.q_table_a)+len(agent.q_table_b)} | ε: {agent.epsilon:.3f}")
            wins = losses = timeouts = 0
    
    return agent


def _self_play_opponent(agent: QLearningAgent):
    """Return an opponent policy that reuses the same agent for self-play games."""
    def policy(game: GameState):
        # Temporarily disable exploration for the opponent side to avoid noisy play
        was_training = agent.training
        agent.training = False
        move = agent.choose_action(game)
        agent.training = was_training
        return move
    return policy


def _random_opponent():
    """Return a random opponent policy for curriculum learning."""
    def policy(game: GameState):
        moves = legal_moves(game)
        return random.choice(moves) if moves else ("Pass", ())
    return policy


def train_agent_mixed(agent: QLearningAgent,
                      n_episodes: int = 2000,
                      max_steps: int = 400,
                      verbose: bool = True,
                      epsilon_floor: float = 0.05,
                      decay_every: int = 500,
                      decay_factor: float = 0.95,
                      history: List[Dict] = None,
                      progress_every: int = 200,
                      curriculum: bool = True):
    """
    Train against opponents with curriculum learning.
    - Starts with random opponent (easy wins) if curriculum=True
    - Gradually increases difficulty: random -> rule-bot -> self-play
    - Decays epsilon periodically down to a floor.
    """
    random_bot = _random_opponent()
    rule_bot = agent_decide_move
    self_play = _self_play_opponent(agent)
    
    agent.training = True
    wins = losses = timeouts = 0

    for episode in range(n_episodes):
        # Curriculum: first 20% random, next 40% rule-bot, last 40% mixed
        if curriculum:
            if episode < n_episodes * 0.2:
                opponent = random_bot
            elif episode < n_episodes * 0.6:
                opponent = rule_bot
            else:
                opponent = self_play if episode % 2 == 0 else rule_bot
        else:
            # Original: alternate rule-bot and self-play
            opponent = rule_bot if episode % 2 == 0 else self_play
        
        game = setup_game(num_players=2)

        for step in range(max_steps):
            game = resolve_plus2(game)
            idx = game.active_idx

            state = get_state_features(game, 0)

            # Choose action
            if idx == 0:
                move = agent.choose_action(game)
            else:
                move = opponent(game)

            move_type, payload = move
            action = get_action_features(move_type, payload, game, idx)

            game_before = game
            game = apply_move(game, move_type, payload)

            if move_type != "PlayAction" or payload[0].action != "AUSSETZEN":
                game.active_idx = next_player_index(game, steps=1)

            game, winner = end_if_win(game)

            # Update only when agent is the mover
            if idx == 0:
                reward = compute_reward(game_before, game, move, 0, winner)
                next_state = get_state_features(game, 0)
                next_moves = legal_moves(game) if winner is None else []
                agent.update(state, action, reward, next_state, next_moves, game, winner is not None)

            if winner is not None:
                if winner == 0:
                    wins += 1
                elif winner == 1:
                    losses += 1
                else:
                    timeouts += 1
                break
        else:
            timeouts += 1
            # Timeout penalty
            agent.update(state, action, REWARDS["timeout"], state, [], game, True)

        # Epsilon schedule
        if (episode + 1) % decay_every == 0 and agent.epsilon > epsilon_floor:
            agent.epsilon = max(epsilon_floor, agent.epsilon * decay_factor)

        # Progress report
        if (episode + 1) % progress_every == 0:
            total = wins + losses + timeouts
            win_rate = wins / total if total else 0.0
            loss_rate = losses / total if total else 0.0
            timeout_rate = timeouts / total if total else 0.0
            if history is not None:
                history.append({
                    "episode": episode + 1,
                    "win_rate": win_rate,
                    "loss_rate": loss_rate,
                    "timeout_rate": timeout_rate,
                    "epsilon": agent.epsilon,
                    "q_size": len(agent.q_table_a) + len(agent.q_table_b),
                })
            if verbose:
                print(f"[Mixed] Episode {episode+1:4d} | Win: {win_rate:5.1%} | Loss: {loss_rate:5.1%} | "
                      f"Timeout: {timeout_rate:5.1%} | Q-size: {len(agent.q_table_a)+len(agent.q_table_b)} | ε: {agent.epsilon:.3f}")
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


def plot_training_curves(mixed_history: List[Dict], focused_history: List[Dict], mixed_episodes: int):
    """Plot and save training win-rate curves from history blocks."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available; skipping plot. Install with: pip install matplotlib")
        return
    
    if not mixed_history and not focused_history:
        print("No training history to plot.")
        return

    plt.figure(figsize=(10, 5))

    if mixed_history:
        plt.plot(
            [h["episode"] for h in mixed_history],
            [h["win_rate"] for h in mixed_history],
            label="Mixed (block win rate)",
        )

    if focused_history:
        plt.plot(
            [mixed_episodes + h["episode"] for h in focused_history],
            [h["win_rate"] for h in focused_history],
            label="Focused (block win rate)",
        )

    plt.xlabel("Episode")
    plt.ylabel("Win rate (block)")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_winrate.png")
    plt.close()
    print("Saved training curve to training_winrate.png")


def print_training_table(mixed_history: List[Dict], focused_history: List[Dict], mixed_episodes: int):
    """Print training history as a formatted table."""
    print("\n" + "=" * 90)
    print("TRAINING PROGRESS TABLE")
    print("=" * 90)
    print(f"{'Phase':<10} | {'Episode':>8} | {'Win Rate':>10} | {'Loss Rate':>10} | {'Timeout':>10} | {'Epsilon':>8} | {'Q-size':>8}")
    print("-" * 90)
    
    if mixed_history:
        for h in mixed_history:
            print(
                f"{'Mixed':<10} | {h['episode']:>8} | {h['win_rate']:>9.1%} | {h['loss_rate']:>9.1%} | "
                f"{h['timeout_rate']:>9.1%} | {h['epsilon']:>7.3f} | {h['q_size']:>8}"
            )
    
    if focused_history:
        for h in focused_history:
            ep = mixed_episodes + h["episode"]
            print(
                f"{'Focused':<10} | {ep:>8} | {h['win_rate']:>9.1%} | {h['loss_rate']:>9.1%} | "
                f"{h['timeout_rate']:>9.1%} | {h['epsilon']:>7.3f} | {h['q_size']:>8}"
            )
    
    print("=" * 90)


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
    print("=" * 60)
    print("DOUBLE Q-LEARNING AGENT TRAINING")
    print("Reduces overestimation bias for more stable learning")
    print("=" * 60)
    print_reward_table()

    # Hyperparameters optimized for Double Q-Learning
    alpha = 0.3        # Higher learning rate (Double Q is more stable)
    gamma = 0.99       # Very high discount for long-term strategy
    epsilon = 0.4      # Higher initial exploration (reduced bias allows more)
    optimistic_init = 20.0  # Higher optimism (Double Q handles it better)
    replay_size = 20000     # Larger replay buffer

    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, 
                          optimistic_init=optimistic_init, replay_size=replay_size)

    mixed_episodes = 3000   # Increased for curriculum learning phases
    focused_episodes = 1200 # More refinement
    mixed_history: List[Dict] = []
    focused_history: List[Dict] = []

    print("\nPhase 1: Curriculum learning (random -> rule-bot -> self-play)")
    train_agent_mixed(
        agent,
        n_episodes=mixed_episodes,
        max_steps=400,
        verbose=True,
        epsilon_floor=0.02,  # Very low floor for strong exploitation
        decay_every=200,     # Aggressive decay
        decay_factor=0.96,   # Faster convergence to exploitation
        history=mixed_history,
        curriculum=True,     # Enable curriculum learning
    )

    print("\nPhase 2: Focused training vs rule-bot (polish)")
    train_agent(
        agent,
        agent_decide_move,
        n_episodes=focused_episodes,
        max_steps=400,
        verbose=True,
        history=focused_history,
    )

    print_training_table(mixed_history, focused_history, mixed_episodes)
    plot_training_curves(mixed_history, focused_history, mixed_episodes)

    # Evaluate with no exploration
    print("\n" + "=" * 75)
    print("Evaluation (500 games, no exploration)")
    print("=" * 75)
    agent.training = False
    agent.epsilon = 0.0
    results = evaluate_agent(agent, agent_decide_move, n_games=500)
    print(
        f"RL Agent vs Rule-bot | Win: {results['win_rate']:5.1%} | "
        f"Loss: {results['losses']/500:5.1%} | Timeout: {results['timeouts']/500:5.1%} | "
        f"Steps: {results['avg_steps']:.0f}"
    )

    # Save model
    agent.save("rl_agent_mixed.pkl")
