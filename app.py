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
    next_player_index, agent_decide_move, Card, GameState
)
from rl_agent import QLearningAgent

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
socketio = SocketIO(app, cors_allowed_origins="*")

# Store active game sessions
games: Dict[str, Dict[str, Any]] = {}

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
        'rl_agent': rl,
        'state': state,
        'created_at': datetime.now(),
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
        'current_turn': 'player' if state.active_idx == 0 else 'opponent'
    }

def play_bot_until_player(game: Dict[str, Any]):
    state: GameState = game['state']
    agent_type = game['agent_type']
    rl = game.get('rl_agent')
    winner = None
    # Let bot play until it's player's turn or game ends
    loop_guard = 0
    while state.active_idx == 1 and winner is None and loop_guard < 20:
        if agent_type == 'rl' and rl is not None:
            move = rl.choose_action(state)
        else:
            move = agent_decide_move(state)
        state = apply_move(state, move[0], move[1])
        # Advance turn unless AUSSETZEN already advanced inside
        if move[0] != 'PlayAction' or (move[0] == 'PlayAction' and move[1][0].action != 'AUSSETZEN'):
            state.active_idx = next_player_index(state, steps=1)
        state, winner = end_if_win(state)
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
            'rl_agent': None,
            'state': state,
            'created_at': datetime.now(),
        }
        game = games[game_id]
    join_room(game_id)
    emit('game_state', serialize_state(game['state']))

@socketio.on('draw_card')
def draw_card():
    # Identify the game via rooms the sid is in
    # Simple approach: pick most recent game
    for gid, game in games.items():
        state: GameState = game['state']
        if state.active_idx != 0:
            continue
        # Apply draw
        state = apply_move(state, 'Draw', ())
        # Advance turn
        state.active_idx = next_player_index(state, steps=1)
        state, winner = end_if_win(state)
        game['state'] = state
        # Bot plays
        winner = play_bot_until_player(game)
        emit('game_state', serialize_state(game['state']), to=gid)
        if winner is not None:
            emit('game_over', {'winner': 'player' if winner == 0 else 'opponent'}, to=gid)
        return

@socketio.on('play_card')
def play_card(data):
    card_index = data.get('card_index')
    # Find current game where it's player's turn
    for gid, game in games.items():
        state: GameState = game['state']
        if state.active_idx != 0:
            continue
        player = state.players[0]
        if card_index is None or card_index < 0 or card_index >= len(player.hand):
            emit('game_state', serialize_state(state), to=gid)
            return
        c = player.hand[card_index]
        # Determine move type for single-card play
        mt = 'PlayNumber' if c.kind == 'NUMBER' else 'PlayAction'
        payload = (c,)
        # Special: TAUSCH must target number stack by default
        if mt == 'PlayAction' and c.action == 'TAUSCH':
            payload = (c, 'number')
        state = apply_move(state, mt, payload)
        # Advance turn
        if not (mt == 'PlayAction' and c.action == 'AUSSETZEN'):
            state.active_idx = next_player_index(state, steps=1)
        state, winner = end_if_win(state)
        game['state'] = state
        # Bot plays
        winner = play_bot_until_player(game)
        emit('game_state', serialize_state(game['state']), to=gid)
        if winner is not None:
            emit('game_over', {'winner': 'player' if winner == 0 else 'opponent'}, to=gid)
        return

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
