// Game board JavaScript
const socket = io();

// Get player info from session storage
const playerName = sessionStorage.getItem('playerName') || 'Player';
const agentType = sessionStorage.getItem('agentType') || 'rule_based';

// DOM Elements
const playerNameEl = document.getElementById('playerName');
const opponentNameEl = document.getElementById('opponentName');
const playerHandEl = document.getElementById('playerHand');
const opponentHandEl = document.getElementById('opponentHand');
const discardPileEl = document.getElementById('discardPile');
const actionDiscardEl = document.getElementById('actionDiscard');
const deckCountEl = document.getElementById('deckCount');
const statusMessageEl = document.getElementById('statusMessage');
const turnIndicatorEl = document.getElementById('turnIndicator');

// Code slot elements
const codeSlots = [
    document.getElementById('code0'),
    document.getElementById('code1'),
    document.getElementById('code2'),
    document.getElementById('code3')
];

// Removed guess inputs; gameplay is draw/drop until code achieved

// Game state
let gameState = {
    player_hand: [],
    opponent_hand_count: 0,
    player_code: null,
    deck_count: 0,
    current_turn: null,
    discard_top: null
};

// Initialize
if (playerNameEl) playerNameEl.textContent = playerName;
if (opponentNameEl) opponentNameEl.textContent = agentType === 'rl' ? 'RL Agent' : 'Rule-Based Agent';

// Socket events
socket.on('connect', () => {
    console.log('Connected to game server');
    const gameId = sessionStorage.getItem('gameId');
    socket.emit('join_game', {
        game_id: gameId,
        player_name: playerName,
        agent_type: agentType
    });
});

socket.on('game_state', (state) => {
    console.log('Received game state:', state);
    updateGameState(state);
});

// Turn updates folded into game_state for simplicity

// No guessing mechanic in CODE

socket.on('game_over', (data) => {
    console.log('Game over:', data);
    handleGameOver(data);
});

// Update game state
function updateGameState(state) {
    gameState = { ...gameState, ...state };
    
    // Update player's code
    if (state.player_code) {
        state.player_code.forEach((digit, index) => {
            codeSlots[index].textContent = digit;
            if (digit !== '?') {
                codeSlots[index].classList.add('has-card');
            } else {
                codeSlots[index].classList.remove('has-card');
            }
        });
    }
    
    // Update player's hand
    if (state.player_hand) {
        renderPlayerHand(state.player_hand);
    }
    
    // Update opponent's hand (card backs)
    if (state.opponent_hand_count !== undefined) {
        renderOpponentHand(state.opponent_hand_count);
    }
    
    // Update deck count
    if (state.deck_count !== undefined) {
        deckCountEl.textContent = state.deck_count;
    }
    
    // Update discard pile
    if (state.discard_top) {
        renderDiscardPile(state.discard_top);
    }

    // Action discard pile (stack last few action cards)
    if (state.action_discard && Array.isArray(state.action_discard)) {
        renderActionDiscardPile(state.action_discard);
    }

    // Update turn indicator
    if (state.current_turn) {
        updateTurn({ current_turn: state.current_turn });
    }
}

// Render player's hand with sorting and overlapping
function renderPlayerHand(cards) {
    playerHandEl.innerHTML = '';
    
    // Sort cards: number cards by value (low to high), then action cards
    const sortedCards = [...cards].sort((a, b) => {
        if (a.type === 'number' && b.type === 'number') {
            return a.value - b.value;
        }
        if (a.type === 'number') return -1;
        if (b.type === 'number') return 1;
        return 0;
    });
    
    const cardCount = sortedCards.length;
    const maxSpreadDeg = 22; // total spread angle limit
    const maxOffset = 38; // horizontal offset per card

    const spreadDeg = cardCount > 1 ? Math.min(maxSpreadDeg, (cardCount - 1) * 5) : 0;
    const angleStep = cardCount > 1 ? spreadDeg / (cardCount - 1) : 0;
    const startAngle = -spreadDeg / 2;

    const totalWidth = Math.min(maxOffset * (cardCount - 1), 400);
    const baseOffset = cardCount > 1 ? totalWidth / (cardCount - 1) : 0;

    sortedCards.forEach((card, index) => {
        const cardEl = createCardElement(card, index);
        const angle = startAngle + angleStep * index;
        const xOffset = -totalWidth / 2 + baseOffset * index;
        const raise = Math.abs(angle) * 0.6; // edges lower, center higher

        cardEl.style.setProperty('--card-offset', `${xOffset}px`);
        cardEl.style.setProperty('--card-rotation', `${angle}deg`);
        cardEl.style.setProperty('--card-raise', `${raise}px`);
        cardEl.style.zIndex = index + 1;

        playerHandEl.appendChild(cardEl);
    });
}

// Render opponent's hand (card backs)
function renderOpponentHand(count) {
    if (!opponentHandEl) return;
    opponentHandEl.innerHTML = '';
    const capped = Math.min(count || 0, 10);
    // Mirror player's fan constants and behavior
    const maxSpreadDeg = 22;
    const maxOffset = 38;
    const spreadDeg = capped > 1 ? Math.min(maxSpreadDeg, (capped - 1) * 5) : 0;
    const angleStep = capped > 1 ? spreadDeg / (capped - 1) : 0;
    const startAngle = -spreadDeg / 2;
    const totalWidth = Math.min(maxOffset * (capped - 1), 400);
    const baseOffset = capped > 1 ? totalWidth / (capped - 1) : 0;

    for (let i = 0; i < capped; i++) {
        const angle = startAngle + angleStep * i;
        const xOffset = -totalWidth / 2 + baseOffset * i;
        // Edges lower than center; boost curvature slightly for opponent
        const raise = Math.abs(angle) * 0.9;
        const card = document.createElement('div');
        card.className = 'opponent-card';
        card.style.setProperty('--card-z', String(i + 1));
        card.style.transform = `translateX(${xOffset}px) translateY(${raise}px) rotate(${angle}deg)`;
        card.innerHTML = `<img src="/static/assets/cards/back/deck2_back.png" alt="Gegner Karte">`;
        opponentHandEl.appendChild(card);
    }

    const opponentCount = document.getElementById('opponentCount');
    if (opponentCount) {
        opponentCount.textContent = `${count || 0}`;
    }
}

// Create card element
function createCardElement(card, index) {
    const cardEl = document.createElement('div');
    cardEl.className = 'hand-card';
    cardEl.dataset.index = index;
    
    // Card image path based on card type
    const imagePath = getCardImagePath(card);
    cardEl.innerHTML = `<img src="${imagePath}" alt="${card.type}">`;
    
    // Make card clickable to play
    cardEl.addEventListener('click', () => playCard(index));
    
    return cardEl;
}

// Get card image path
function getCardImagePath(card) {
    // TODO: Implement proper card image path logic based on card type
    // For now, return placeholder
    if (card.type === 'number') {
        const color = card.color.toLowerCase();
        return `/static/assets/cards/number/${color}_${card.value}_copy1.png`;
    } else if (card.type === 'action') {
        return `/static/assets/cards/action/${card.action}_1.png`;
    }
    return '/static/assets/cards/back/deck2_back.png';
}

// Render discard pile
function renderDiscardPile(card) {
    const imagePath = getCardImagePath(card);
    discardPileEl.innerHTML = `<img src="${imagePath}" alt="Discard">`;
}

// Render action discard pile (stack last few)
function renderActionDiscardPile(cards) {
    if (!actionDiscardEl) return;
    actionDiscardEl.innerHTML = '';
    const recent = cards.slice(-4);
    recent.forEach((card) => {
        const cardEl = document.createElement('div');
        cardEl.className = 'action-card';
        const imagePath = getCardImagePath(card);
        cardEl.innerHTML = `<img src="${imagePath}" alt="Action Discard">`;
        actionDiscardEl.appendChild(cardEl);
    });
}

// Play a card
function playCard(cardIndex) {
    if (gameState.current_turn !== 'player') {
        statusMessageEl.textContent = "It's not your turn!";
        return;
    }
    
    socket.emit('play_card', { card_index: cardIndex });
}

// Update turn
function updateTurn(data) {
    gameState.current_turn = data.current_turn;
    
    if (data.current_turn === 'player') {
        if (turnIndicatorEl) turnIndicatorEl.textContent = 'Dein Zug';
        if (statusMessageEl) statusMessageEl.textContent = 'Karte spielen oder aufnehmen';
    } else {
        if (turnIndicatorEl) turnIndicatorEl.textContent = 'Zug des Gegners';
        if (statusMessageEl) statusMessageEl.textContent = 'Warten auf Gegner...';
    }
}

// Click on draw pile image triggers draw
const drawPileEl = document.getElementById('drawPile');
if (drawPileEl) {
    drawPileEl.addEventListener('click', () => {
        if (gameState.current_turn !== 'player') {
            if (statusMessageEl) statusMessageEl.textContent = 'Nicht dein Zug!';
            return;
        }
        socket.emit('draw_card', {});
    });
}

// Handle game over
function handleGameOver(data) {
    const message = data.winner === 'player' 
        ? '🎉 Sieg! Du hast deinen CODE erreicht!' 
        : '😔 Niederlage. Der Gegner war schneller.';
    
    if (statusMessageEl) statusMessageEl.textContent = message;
    if (turnIndicatorEl) turnIndicatorEl.textContent = 'Game Over';
    
    // Show play again button after delay
    setTimeout(() => {
        if (confirm(message + '\n\nNochmal spielen?')) {
            window.location.href = '/';
        }
    }, 2000);
}
