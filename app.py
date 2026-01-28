"""
Flask web application for CODE card game.
Allows players to compete against AI agents (rule-based or RL-trained).
"""

from flask import Flask, render_template, session, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import secrets
from datetime import datetime
import os
from typing import Dict, Any

# Game logic imports
from code_agent_v1 import (
    setup_game, legal_moves, apply_move, end_if_win, is_win,
    next_player_index, agent_decide_move, Card, GameState,
    can_card_be_played, find_playable_moves_for_card
)
from rl_agent import QLearningAgent, get_state_features, get_action_features, compute_reward

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
socketio = SocketIO(app, cors_allowed_origins="*")

# Store active game sessions
games: Dict[str, Dict[str, Any]] = {}
# Map each connected client to its game id for targeted updates
client_to_game: Dict[str, str] = {}
# Track games where player is deciding on drawn card play
drawn_card_pending: Dict[str, Card] = {}

@app.route('/')
def index():
    """Main menu page."""
    return render_template('index.html')

@app.route('/rules')
def rules():
    """Game rules and how to play."""
    return render_template('rules.html')

@app.route('/game')
def game():
    """Game board page."""
    return render_template('game.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f'Client disconnected: {request.sid}')
    # Cleanup client-to-game mapping
    try:
        client_to_game.pop(request.sid, None)
    except Exception:
        pass

@socketio.on('start_game')
def handle_start_game(data):
    """Initialize a new game session."""
    player_name = data.get('player_name', 'Player')
    agent_type = data.get('agent_type', 'rule_based')  # 'rule_based' or 'rl'
    
    # Initialize game logic (2 players: human vs agent)
    game_id = secrets.token_hex(8)
    state = setup_game(num_players=2)
    rl = None
    if agent_type == 'rl':
        rl = QLearningAgent()
        rl.training = False
        # Try load model if available
        try:
            rl.load(os.path.join(os.path.dirname(__file__), 'rl_agent_model.pkl'))
        except Exception:
            pass

    games[game_id] = {
        'player_name': player_name,
        'agent_type': agent_type,
        'training_mode': data.get('training_mode', False),  # Enable human training
        'rl_agent': rl,
        'state': state,
        'created_at': datetime.now(),
        'move_history': [],  # Track moves for RL learning: [(state_before, move, ai_player_idx, step), ...]
        'total_steps': 0,  # Track game length for efficiency reward
    }
    
    emit('game_started', {
        'game_id': game_id,
        'player_name': player_name,
        'agent_type': agent_type
    })

def serialize_card(c: Card) -> Dict[str, Any]:
    if c.kind == "NUMBER":
        return {"type": "number", "value": c.value, "color": c.color}
    if c.kind == "ACTION":
        return {"type": "action", "action": c.action}
    return {"type": "unknown"}

def serialize_state(state: GameState) -> Dict[str, Any]:
    player = state.players[0]
    opponent = state.players[1]
    top_num = state.number_discard[-1] if state.number_discard else None
    return {
        'player_hand': [serialize_card(c) for c in player.hand],
        'opponent_hand_count': len(opponent.hand),
        'player_code': list(player.code.target),
        'deck_count': len(state.draw_pile),
        'discard_top': serialize_card(top_num) if top_num else None,
        'number_discard': [serialize_card(c) for c in state.number_discard],
        'action_discard': [serialize_card(c) for c in state.action_discard],
        'current_turn': 'player' if state.active_idx == 0 else 'opponent',
        'pending_plus2': state.pending_plus2
    }

def ensure_game_fields(game: Dict[str, Any]) -> None:
    """Ensure all required fields exist in game dictionary."""
    if 'total_steps' not in game:
        game['total_steps'] = 0
    if 'move_history' not in game:
        game['move_history'] = []
    if 'training_mode' not in game:
        game['training_mode'] = False

def play_bot_until_player(game: Dict[str, Any], gid: str, step_delay: float = 0.6):
    state: GameState = game['state']
    agent_type = game['agent_type']
    rl = game.get('rl_agent')
    
    # Ensure all required fields exist
    ensure_game_fields(game)
    
    winner = None
    # Let bot play until it's player's turn or game ends
    loop_guard = 0
    while state.active_idx == 1 and winner is None and loop_guard < 20:
        # Handle PLUS2 penalty: draw multiple times and skip further play
        if state.pending_plus2 > 0:
            draws = state.pending_plus2
            for _ in range(draws):
                state = apply_move(state, 'Draw', ())
            state.pending_plus2 = 0
            # Advance turn after forced draws
            state.active_idx = next_player_index(state, steps=1)
        else:
            # Track state before move for learning
            state_features_before = None
            if agent_type == 'rl' and rl is not None:
                state_features_before = get_state_features(state, 1)  # Player 1 (AI)
            
            # Normal move selection
            if agent_type == 'rl' and rl is not None:
                move = rl.choose_action(state)
            else:
                move = agent_decide_move(state)
            
            # Track GESCHENK data before applying move
            geschenk_data = None
            if move[0] == 'PlayAction' and len(move[1]) > 0 and hasattr(move[1][0], 'action') and move[1][0].action == 'GESCHENK':
                # Bot is playing GESCHENK, find which card it will take from player
                target_idx = move[1][1] if len(move[1]) > 1 else 0
                if target_idx == 0:  # Taking from player
                    player = state.players[0]
                    # Determine which card will be taken
                    if len(move[1]) > 2:
                        idx = move[1][2]
                        if 0 <= idx < len(player.hand):
                            taken_card = player.hand[idx]
                    else:
                        from code_agent_v1 import choose_best_gift
                        taken_card = choose_best_gift(state, 0)
                    if taken_card:
                        geschenk_data = {
                            'taken_card': serialize_card(taken_card)
                        }
            
            state = apply_move(state, move[0], move[1])
            # Advance turn unless AUSSETZEN already advanced inside
            if move[0] != 'PlayAction' or (move[0] == 'PlayAction' and move[1][0].action != 'AUSSETZEN'):
                state.active_idx = next_player_index(state, steps=1)
            
            # Store move for learning
            if agent_type == 'rl' and rl is not None and state_features_before is not None:
                # Ensure these fields exist before accessing
                if 'move_history' not in game:
                    game['move_history'] = []
                if 'total_steps' not in game:
                    game['total_steps'] = 0
                
                game['move_history'].append({
                    'state_before': state_features_before,
                    'move': move,
                    'step': game['total_steps'],
                })
        
        state, winner = end_if_win(state)
        game['state'] = state
        
        # Ensure total_steps exists before incrementing
        if 'total_steps' not in game:
            game['total_steps'] = 0
        game['total_steps'] += 1
        # Emit step update to allow client-side animation
        try:
            # Generate move description for client notification
            move_desc = describe_move_for_client(move[0], move[1], None)
            opponent_move_data = {
                'move_type': move[0],
                'move_description': f'Gegner spielte: {move_desc}',
                'move_card': serialize_card(move[1][0]) if move and move[1] else None,
                'state': serialize_state(state)
            }
            if geschenk_data:
                opponent_move_data['geschenk_data'] = geschenk_data
            emit('opponent_move', opponent_move_data, to=gid)
            # Non-blocking sleep for Socket.IO workers
            socketio.sleep(step_delay)
        except Exception:
            pass
        loop_guard += 1
    game['state'] = state
    return winner

@socketio.on('join_game')
def join_game(data):
    game_id = data.get('game_id')
    player_name = data.get('player_name', 'Player')
    agent_type = data.get('agent_type', 'rule_based')
    game = None
    if game_id and game_id in games:
        game = games[game_id]
    else:
        # Fallback: create a new game if missing
        game_id = secrets.token_hex(8)
        state = setup_game(num_players=2)
        games[game_id] = {
            'player_name': player_name,
            'agent_type': agent_type,
            'training_mode': False,
            'rl_agent': None,
            'state': state,
            'created_at': datetime.now(),
            'move_history': [],
            'total_steps': 0,
        }
        game = games[game_id]
    join_room(game_id)
    # Track which game this client is in
    client_to_game[request.sid] = game_id
    # If it's the opponent's turn when the client joins, let the bot
    # play until it's the player's turn so the UI updates immediately.
    winner = None
    if game['state'].active_idx == 1:
        winner = play_bot_until_player(game, game_id)
    emit('game_state', serialize_state(game['state']))
    if winner is not None:
        # Learn from game if RL agent
        if game['agent_type'] == 'rl' and game.get('rl_agent') is not None:
            learn_from_game(game, winner)
        
        # Reveal opponent details on win
        if winner == 1:
            opp = game['state'].players[1]
            emit('game_over', {
                'winner': 'opponent',
                'opponent_hand': [serialize_card(c) for c in opp.hand],
                'opponent_code': list(opp.code.target)
            }, to=game_id)
        else:
            emit('game_over', {'winner': 'player'}, to=game_id)

@socketio.on('draw_card')
def draw_card(data=None):
    # Identify the game from payload or mapping
    gid = (data or {}).get('game_id') if isinstance(data, dict) else None
    if not gid:
        gid = client_to_game.get(request.sid)
    print(f"draw_card called, gid={gid}")
    if not gid or gid not in games:
        print(f"Game not found for gid={gid}")
        return
    game = games[gid]
    ensure_game_fields(game)
    state: GameState = game['state']
    if state.active_idx != 0:
        # Not player's turn; just send current state back
        print(f"Not player's turn, active_idx={state.active_idx}")
        emit('game_state', serialize_state(state), to=gid)
        return
    
    # Track the drawn card for potential immediate play
    drawn_card = None
    
    # Handle PLUS2: draw multiple times if pending
    if state.pending_plus2 > 0:
        draws = state.pending_plus2
        for i in range(draws):
            state = apply_move(state, 'Draw', ())
            # Store the last drawn card for checking immediate play
            if i == draws - 1:
                drawn_card = state.players[0].hand[-1]
        state.pending_plus2 = 0
    else:
        # Normal single draw
        state = apply_move(state, 'Draw', ())
        drawn_card = state.players[0].hand[-1]
    
    print(f"\n=== DREW CARD: {drawn_card} ===")
    
    # Check if the drawn card can be played immediately
    playable_moves = find_playable_moves_for_card(drawn_card, state) if drawn_card else []
    print(f"Found {len(playable_moves)} playable moves")
    
    # Only pause and emit modal if card is playable
    if playable_moves:
        # Card is playable - pause game and let player decide
        game['state'] = state
        drawn_card_pending[gid] = drawn_card
        
        print(f"Emitting game_state to room {gid}")
        emit('game_state', serialize_state(state), to=gid)
        
        print(f"Emitting card_drawn event with play options")
        available_moves_data = [
            {'move_type': mt, 'has_payload': len(p) > 0} 
            for mt, p in playable_moves
        ]
        
        emit('card_drawn', {
            'card': serialize_card(drawn_card),
            'is_playable': True,
            'available_moves': available_moves_data
        }, to=gid)
        print(f"=== WAITING FOR PLAYER DECISION ===\n")
        return
    
    # Card is NOT playable - advance turn normally
    print("Card not playable - advancing turn")
    state.active_idx = next_player_index(state, steps=1)
    state, winner = end_if_win(state)
    game['state'] = state
    # Bot plays
    winner = play_bot_until_player(game, gid)
    emit('game_state', serialize_state(game['state']), to=gid)
    if winner is not None:
        # Learn from game if RL agent
        learn_from_game(game, winner)
        
        if winner == 1:
            opp = game['state'].players[1]
            emit('game_over', {
                'winner': 'opponent',
                'opponent_hand': [serialize_card(c) for c in opp.hand],
                'opponent_code': list(opp.code.target)
            }, to=gid)
        else:
            emit('game_over', {'winner': 'player'}, to=gid)

@socketio.on('play_drawn_card')
def play_drawn_card(data=None):
    """Handle immediate play of a drawn card."""
    data = data or {}
    gid = data.get('game_id') or client_to_game.get(request.sid)
    if not gid or gid not in games:
        return
    game = games[gid]
    state: GameState = game['state']
    
    # Clear pending drawn card flag
    drawn_card_pending.pop(gid, None)
    
    if state.active_idx != 0:
        emit('game_state', serialize_state(state), to=gid)
        return
    
    move_type = data.get('move_type')
    card = data.get('card')
    
    if not move_type or not card:
        emit('game_state', serialize_state(state), to=gid)
        return
    
    player = state.players[0]
    # Find the drawn card in player's hand
    drawn_card = None
    for c in player.hand:
        if serialize_card(c) == card:
            drawn_card = c
            break
    
    if not drawn_card:
        emit('game_state', serialize_state(state), to=gid)
        return
    
    # Handle the move based on type
    if move_type == 'PlayNumber':
        state = apply_move(state, 'PlayNumber', (drawn_card,))
    elif move_type == 'PlayAction':
        state = apply_move(state, 'PlayAction', (drawn_card,))
    else:
        emit('game_state', serialize_state(state), to=gid)
        return
    
    # Advance turn (but not for AUSSETZEN, which handles its own skip logic)
    if not (move_type == 'PlayAction' and hasattr(drawn_card, 'action') and drawn_card.action == 'AUSSETZEN'):
        state.active_idx = next_player_index(state, steps=1)
    state, winner = end_if_win(state)
    game['state'] = state
    
    # Bot plays
    winner = play_bot_until_player(game, gid)
    emit('game_state', serialize_state(game['state']), to=gid)
    
    if winner is not None:
        if winner == 1:
            opp = game['state'].players[1]
            emit('game_over', {
                'winner': 'opponent',
                'opponent_hand': [serialize_card(c) for c in opp.hand],
                'opponent_code': list(opp.code.target)
            }, to=gid)
        else:
            emit('game_over', {'winner': 'player'}, to=gid)


@socketio.on('drawn_card_add_to_hand')
def drawn_card_add_to_hand(data=None):
    """Player decides not to play the drawn card immediately."""
    data = data or {}
    gid = data.get('game_id') or client_to_game.get(request.sid)
    if not gid or gid not in games:
        return
    
    # Clear pending drawn card flag
    drawn_card_pending.pop(gid, None)
    
    game = games[gid]
    ensure_game_fields(game)
    state: GameState = game['state']
    
    # Advance turn (drawn card stays in hand)
    state.active_idx = next_player_index(state, steps=1)
    state, winner = end_if_win(state)
    game['state'] = state
    
    # Bot plays
    winner = play_bot_until_player(game, gid)
    emit('game_state', serialize_state(game['state']), to=gid)
    
    if winner is not None:
        # Learn from game if RL agent
        learn_from_game(game, winner)
        
        if winner == 1:
            opp = game['state'].players[1]
            emit('game_over', {
                'winner': 'opponent',
                'opponent_hand': [serialize_card(c) for c in opp.hand],
                'opponent_code': list(opp.code.target)
            }, to=gid)
        else:
            emit('game_over', {'winner': 'player'}, to=gid)


@socketio.on('play_card')
def play_card(data=None):
    data = data or {}
    card_index = data.get('card_index')
    target_opponent = data.get('target_opponent')  # For RESET/GESCHENK
    tausch_target = data.get('target')  # For TAUSCH: 'number' or 'action'
    gift_index = data.get('gift_index')  # For GESCHENK: opponent card index to take
    
    print(f"\n=== PLAY_CARD RECEIVED ===")
    print(f"card_index: {card_index}")
    print(f"target_opponent: {target_opponent}")
    print(f"gift_index: {gift_index}")
    print(f"tausch_target: {tausch_target}")
    print(f"==========================\n")
    
    gid = data.get('game_id') or client_to_game.get(request.sid)
    if not gid or gid not in games:
        return
    game = games[gid]
    ensure_game_fields(game)
    state: GameState = game['state']
    if state.active_idx != 0:
        emit('game_state', serialize_state(state), to=gid)
        return
    player = state.players[0]
    if card_index is None or card_index < 0 or card_index >= len(player.hand):
        emit('game_state', {**serialize_state(state), 'status': 'Ungültige Auswahl.'}, to=gid)
        return
    c = player.hand[card_index]
    # Enforce legality: pick matching legal move for the chosen card only
    legals = legal_moves(state)
    chosen = None
    
    # Find matching legal move
    for mt, payload in legals:
        if mt in ('PlayNumber', 'PlayAny') and payload and payload[0] is c:
            chosen = (mt, payload)
            break
        if mt == 'PlayAction' and payload and payload[0] is c:
            # For action cards that need special parameters (RESET, GESCHENK, TAUSCH)
            if c.action == 'TAUSCH':
                # TAUSCH: match the target if provided
                if tausch_target and len(payload) > 1 and payload[1] == tausch_target:
                    chosen = (mt, payload)
                    break
                elif not tausch_target and len(payload) > 1:
                    # Accept first available if no target specified
                    chosen = (mt, payload)
                    break
            elif c.action in ('RESET', 'GESCHENK'):
                # Check if this legal move matches the target we're sending
                if target_opponent is not None and len(payload) > 1 and payload[1] == target_opponent:
                    chosen = (mt, payload)
                    break
            else:
                # Simple action cards (RICHTUNGSWECHSEL, AUSSETZEN, PLUS2)
                chosen = (mt, payload)
                break
    
    if chosen is None:
        # Not a legal single-card play; reject
        emit('game_state', {**serialize_state(state), 'status': 'Karte nicht spielbar.'}, to=gid)
        return
    mt, payload = chosen
    # If player specified which card to take for GESCHENK, attach it
    if mt == 'PlayAction' and c.action == 'GESCHENK' and target_opponent is not None and gift_index is not None:
        print(f"Adding gift_index {gift_index} to GESCHENK payload")
        print(f"Original payload: {payload}")
        payload = (payload[0], payload[1], int(gift_index))
        print(f"New payload: {payload}")
    state = apply_move(state, mt, payload)
    # Advance turn (but not for AUSSETZEN)
    if not (mt == 'PlayAction' and c.action == 'AUSSETZEN'):
        state.active_idx = next_player_index(state, steps=1)
    state, winner = end_if_win(state)
    game['state'] = state

    # Emit the player's move state before the bot acts so the top card is visible
    emit('game_state', serialize_state(state), to=gid)

    # If the move ended the game, announce now
    if winner is not None:
        # Learn from game if RL agent
        learn_from_game(game, winner)
        
        if winner == 1:
            opp = game['state'].players[1]
            emit('game_over', {
                'winner': 'opponent',
                'opponent_hand': [serialize_card(c) for c in opp.hand],
                'opponent_code': list(opp.code.target)
            }, to=gid)
        else:
            emit('game_over', {'winner': 'player'}, to=gid)
        return

    # Bot plays
    winner = play_bot_until_player(game, gid)
    emit('game_state', serialize_state(game['state']), to=gid)
    if winner is not None:
        # Learn from game if RL agent
        learn_from_game(game, winner)
        
        if winner == 1:
            opp = game['state'].players[1]
            emit('game_over', {
                'winner': 'opponent',
                'opponent_hand': [serialize_card(c) for c in opp.hand],
                'opponent_code': list(opp.code.target)
            }, to=gid)
        else:
            emit('game_over', {'winner': 'player'}, to=gid)

@socketio.on('play_sum')
def play_sum(data=None):
    data = data or {}
    gid = data.get('game_id') or client_to_game.get(request.sid)
    if not gid or gid not in games:
        return
    idx_a = data.get('index_a')
    idx_b = data.get('index_b')
    game = games[gid]
    state: GameState = game['state']
    if state.active_idx != 0:
        emit('game_state', serialize_state(state), to=gid)
        return
    player = state.players[0]
    if None in (idx_a, idx_b):
        emit('game_state', {**serialize_state(state), 'status': 'Bitte zwei Karten wählen.'}, to=gid)
        return
    if idx_a == idx_b or idx_a < 0 or idx_b < 0 or idx_a >= len(player.hand) or idx_b >= len(player.hand):
        emit('game_state', {**serialize_state(state), 'status': 'Ungültige Auswahl.'}, to=gid)
        return
    a = player.hand[idx_a]
    b = player.hand[idx_b]
    # Enforce that selected move exists
    legals = legal_moves(state)
    chosen = None
    for mt, payload in legals:
        if mt == 'PlaySum' and ((payload[0] is a and payload[1] is b) or (payload[0] is b and payload[1] is a)):
            chosen = (mt, payload)
            break
    if chosen is None:
        emit('game_state', {**serialize_state(state), 'status': 'Diese zwei Karten können nicht als Summe gespielt werden.'}, to=gid)
        return
    mt, payload = chosen
    # Preserve the player's selection order: first clicked goes to the bottom
    payload = (a, b)
    state = apply_move(state, mt, payload)
    state.active_idx = next_player_index(state, steps=1)
    state, winner = end_if_win(state)
    game['state'] = state
    # Emit player's sum before bot acts so the two-card stack is visible in order
    emit('game_state', serialize_state(game['state']), to=gid)

    # If the game ended with the player's sum, announce immediately
    if winner is not None:
        # Learn from game if RL agent
        learn_from_game(game, winner)
        
        if winner == 1:
            opp = game['state'].players[1]
            emit('game_over', {
                'winner': 'opponent',
                'opponent_hand': [serialize_card(c) for c in opp.hand],
                'opponent_code': list(opp.code.target)
            }, to=gid)
        else:
            emit('game_over', {'winner': 'player'}, to=gid)
        return

    winner = play_bot_until_player(game, gid)
    emit('game_state', serialize_state(game['state']), to=gid)
    if winner is not None:
        # Learn from game if RL agent
        learn_from_game(game, winner)
        
        if winner == 1:
            opp = game['state'].players[1]
            emit('game_over', {
                'winner': 'opponent',
                'opponent_hand': [serialize_card(c) for c in opp.hand],
                'opponent_code': list(opp.code.target)
            }, to=gid)
        else:
            emit('game_over', {'winner': 'player'}, to=gid)

@socketio.on('get_legal_moves')
def get_legal_moves_handler(data=None):
    data = data or {}
    gid = data.get('game_id') or client_to_game.get(request.sid)
    if not gid or gid not in games:
        return
    game = games[gid]
    state: GameState = game['state']
    if state.active_idx != 0:
        emit('legal_moves', {'moves': []})
        return
    
    # Get legal moves from game logic
    legals = legal_moves(state)
    player = state.players[0]
    
    # Format moves for client
    def index_by_identity(lst, target):
        for i, obj in enumerate(lst):
            if obj is target:
                return i
        return -1
    formatted_moves = []
    for idx, (move_type, payload) in enumerate(legals):
        move_data = {
            'index': idx,
            'type': move_type,
            'description': describe_move_for_client(move_type, payload, player)
        }
        
        # Include card indices for reference
        if move_type in ('PlayNumber', 'PlayAny', 'PlayAction'):
            card = payload[0]
            idx = index_by_identity(player.hand, card)
            if idx != -1:
                move_data['card_index'] = idx
        elif move_type == 'PlaySum':
            a_idx = index_by_identity(player.hand, payload[0])
            b_idx = index_by_identity(player.hand, payload[1])
            if a_idx != -1 and b_idx != -1:
                move_data['card_indices'] = [a_idx, b_idx]
        
        formatted_moves.append(move_data)
    
    emit('legal_moves', {'moves': formatted_moves})


def learn_from_game(game: Dict[str, Any], winner: int):
    """
    Learn from a completed game using RL agent.
    Only learns if agent_type is 'rl' AND training_mode is enabled.
    
    Args:
        game: Game dictionary with move history
        winner: 0 = player won, 1 = AI won, None = timeout
    """
    rl = game.get('rl_agent')
    training_mode = game.get('training_mode', False)
    
    print(f"\n{'='*60}")
    print(f"LEARN_FROM_GAME CALLED")
    print(f"RL agent exists: {rl is not None}")
    print(f"Training mode enabled: {training_mode}")
    print(f"Winner: {winner} (0=player, 1=AI)")
    print(f"Move history length: {len(game.get('move_history', []))}")
    print(f"{'='*60}\n")
    
    if rl is None or not hasattr(rl, 'training') or not training_mode:
        if not training_mode:
            print("⚠️  Training mode NOT enabled - skipping learning")
        return
    
    print(f"🎓 TRAINING STARTED - Learning from {len(game.get('move_history', []))} AI moves...")
    
    # Enable learning mode
    was_training = rl.training
    rl.training = True
    
    # Learn from all AI moves in reverse order
    learned_count = 0
    for move_data in game.get('move_history', []):
        state_before = move_data['state_before']
        move = move_data['move']
        step = move_data.get('step', 0)
        
        # Compute reward (now with step efficiency penalty)
        reward = compute_reward(
            game['state'], game['state'], move, 
            player_idx=1, winner=winner, step=step
        )
        
        # Get action features
        try:
            action_features = get_action_features(move[0], move[1], game['state'], 1)
            next_state = get_state_features(game['state'], 1)
            next_moves = legal_moves(game['state'])
            
            # Update Q-values (learning from human strategy)
            rl.update(state_before, action_features, reward, next_state, next_moves, 
                     game['state'], winner is not None)
            learned_count += 1
        except Exception as e:
            print(f"❌ Error learning from move: {e}")
    
    rl.training = was_training
    
    print(f"✅ Learned from {learned_count} moves")
    
    # Save model after learning from human
    try:
        rl.save(os.path.join(os.path.dirname(__file__), 'rl_agent_model.pkl'))
        print(f"💾 Model saved! Winner: {'Player' if winner == 0 else 'AI' if winner == 1 else 'Timeout'}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"❌ Error saving model: {e}")


@socketio.on('get_opponent_hand')
def get_opponent_hand(data=None):
    data = data or {}
    gid = data.get('game_id') or client_to_game.get(request.sid)
    if not gid or gid not in games:
        return
    game = games[gid]
    state: GameState = game['state']
    # Only allow during player's turn
    if state.active_idx != 0:
        return
    opponent = state.players[1]
    emit('opponent_hand_reveal', {
        'cards': [serialize_card(c) for c in opponent.hand]
    }, to=request.sid)

def describe_move_for_client(move_type: str, payload: tuple, player) -> str:
    """Generate a human-readable description of a move."""
    if move_type == 'Draw':
        return '🃏 Karte aufnehmen'
    
    if move_type in ('PlayNumber', 'PlayAny'):
        card = payload[0]
        color_emoji = {'ROT': '🔴', 'BLAU': '🔵', 'GELB': '🟡', 'VIOLETT': '🟣'}
        emoji = color_emoji.get(card.color, '')
        return f'{emoji} {card.color} {card.value}'
    
    if move_type == 'PlaySum':
        a, b = payload[0], payload[1]
        return f'➕ Summe: {a.value} + {b.value} = {a.value + b.value}'
    
    if move_type == 'PlayAction':
        card = payload[0]
        action_names = {
            'TAUSCH': '🔄 TAUSCH',
            'RICHTUNGSWECHSEL': '↩️ RICHTUNGSWECHSEL',
            'AUSSETZEN': '🚫 AUSSETZEN',
            'PLUS2': '➕2 PLUS2',
            'GESCHENK': '🎁 GESCHENK',
            'RESET': '🔃 RESET'
        }
        return action_names.get(card.action, card.action)
    
    return move_type

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, debug=False, host='0.0.0.0', port=port)
