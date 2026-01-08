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
const playerCodeCardEl = document.getElementById('playerCodeCard');
const movesListEl = document.getElementById('movesList');

// Moves window collapse/expand controls
const movesWindowEl = document.querySelector('.moves-window');
const movesWindowDotsEl = document.querySelector('.moves-window-dots');
const movesSidebarEl = document.querySelector('.moves-sidebar');
if (movesWindowDotsEl && movesWindowEl && movesSidebarEl) {
    movesWindowDotsEl.title = 'Minimieren/Erweitern';
    movesWindowDotsEl.addEventListener('click', () => {
        movesWindowEl.classList.toggle('collapsed');
        movesSidebarEl.classList.toggle('collapsed');
    });
}

// Code window collapse/expand controls
const codeWindowEl = document.querySelector('.code-window');
const codeWindowDotsEl = document.querySelector('.code-window-dots');
const codeSidebarEl = document.querySelector('.code-card-sidebar');
if (codeWindowDotsEl && codeWindowEl && codeSidebarEl) {
    codeWindowDotsEl.title = 'Minimieren/Erweitern';
    codeWindowDotsEl.addEventListener('click', () => {
        codeWindowEl.classList.toggle('collapsed');
        codeSidebarEl.classList.toggle('collapsed');
    });
}

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
    number_discard: [],
    current_turn: null,
    discard_top: null
};

// Track selected indices for potential PlaySum
let selectedIndices = [];
let legalMoves = [];
let sumModeActive = false;
let opponentRevealHand = [];

// Sum mode toggle button
const sumModeToggleBtn = document.getElementById('sumModeToggle');
if (sumModeToggleBtn) {
    sumModeToggleBtn.addEventListener('click', () => {
        sumModeActive = !sumModeActive;
        sumModeToggleBtn.classList.toggle('active', sumModeActive);
        
        // Clear any existing selections when toggling
        selectedIndices = [];
        document.querySelectorAll('.hand-card.selected').forEach(card => {
            card.classList.remove('selected');
        });
        
        // Update status message
        if (statusMessageEl) {
            if (sumModeActive) {
                statusMessageEl.textContent = 'Summen-Modus: Wähle zwei Karten';
            } else {
                statusMessageEl.textContent = gameState.current_turn === 'player' 
                    ? 'Karte spielen oder aufnehmen' 
                    : 'Warten auf Gegner...';
            }
        }
    });
}

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
    requestLegalMoves();
});

socket.on('legal_moves', (data) => {
    legalMoves = data.moves || [];
    renderLegalMoves();
});

// Animate opponent steps and update state progressively
socket.on('opponent_move', (data) => {
    try {
        const state = data.state;
        const moveDescription = data.move_description || 'Gegner hat eine Aktion durchgeführt';
        
        // Show notification
        showOpponentNotification(moveDescription);
        
        // Brief pulse on appropriate pile
        const pile = data.move_type === 'PlayAction' ? actionDiscardEl : discardPileEl;
        if (pile) {
            pile.classList.remove('pulse');
            // Force reflow
            void pile.offsetWidth;
            pile.classList.add('pulse');
            setTimeout(() => pile.classList.remove('pulse'), 400);
        }
        updateGameState(state);
    } catch (e) {
        console.warn('opponent_move handling error', e);
    }
});

// Show opponent move notification
function showOpponentNotification(text) {
    const notification = document.getElementById('opponentNotification');
    if (notification) {
        const textEl = notification.querySelector('.notification-text');
        if (textEl) textEl.textContent = text;
        
        notification.classList.remove('show');
        // Force reflow
        void notification.offsetWidth;
        notification.classList.add('show');
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show');
        }, 3000);
    }
}

function requestOpponentHand() {
    const gid = sessionStorage.getItem('gameId');
    socket.emit('get_opponent_hand', { game_id: gid });
}

// Turn updates folded into game_state for simplicity

// No guessing mechanic in CODE

socket.on('game_over', (data) => {
    console.log('Game over:', data);
    handleGameOver(data);
});

// Receive opponent hand reveal for Geschenk selection
socket.on('opponent_hand_reveal', (data) => {
    opponentRevealHand = data.cards || [];
    populateGeschenkModal();
});

// Update game state
function updateGameState(state) {
    // Track which cards are new (for draw animation) BEFORE updating gameState
    const oldHandLength = (gameState.player_hand || []).length;
    const newHandLength = (state.player_hand || []).length;
    const newIndices = new Set();
    if (newHandLength > oldHandLength) {
        // New cards were added, mark the new ones
        for (let i = oldHandLength; i < newHandLength; i++) {
            newIndices.add(i);
        }
    }
    
    gameState = { ...gameState, ...state };
    if (state.status && statusMessageEl) {
        statusMessageEl.textContent = state.status;
    }
    
    // Update player's code
    if (state.player_code) {
        // Determine which digits are currently in the player's hand (number cards only)
        const digitsInHand = new Set(
            (state.player_hand || [])
                .filter((c) => c.type === 'number')
                .map((c) => String(c.value))
        );

        state.player_code.forEach((digit, index) => {
            codeSlots[index].textContent = digit;
            if (digit !== '?' && digitsInHand.has(String(digit))) {
                codeSlots[index].classList.add('has-card');
            } else {
                codeSlots[index].classList.remove('has-card');
            }
        });

        // Render the player's code card image (no overlay needed)
        try {
            const codeStr = state.player_code.map(String).join('');
            if (playerCodeCardEl && codeStr && codeStr.length === 4) {
                const src = `/static/assets/cards/code/code_${codeStr}.png`;
                playerCodeCardEl.innerHTML = `<img src="${src}" alt="Codekarte ${codeStr}">`;
            }
        } catch (e) {
            // Fail silently if code not ready
        }
    }
    
    // Update player's hand
    if (state.player_hand) {
        renderPlayerHand(state.player_hand, newIndices);
    }
    
    // Update opponent's hand (card backs)
    if (state.opponent_hand_count !== undefined) {
        renderOpponentHand(state.opponent_hand_count);
    }
    
    // Update deck count
    if (state.deck_count !== undefined) {
        deckCountEl.textContent = state.deck_count;
    }
    
    // Update number discard pile
    if (state.number_discard) {
        renderNumberDiscardPile(state.number_discard);
    } else if (state.discard_top) {
        renderNumberDiscardPile([state.discard_top]);
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
function renderPlayerHand(cards, newIndices = new Set()) {
    playerHandEl.innerHTML = '';

    // Keep original indices for server-side mapping
    const withIndex = cards.map((card, origIndex) => ({ card, origIndex }));

    // Sort by number (low→high), then actions
    const sorted = [...withIndex].sort((a, b) => {
        if (a.card.type === 'number' && b.card.type === 'number') {
            return a.card.value - b.card.value;
        }
        if (a.card.type === 'number') return -1;
        if (b.card.type === 'number') return 1;
        return 0;
    });

    const cardCount = sorted.length;
    const maxSpreadDeg = 22; // total spread angle limit
    const maxOffset = 38; // horizontal offset per card

    const spreadDeg = cardCount > 1 ? Math.min(maxSpreadDeg, (cardCount - 1) * 5) : 0;
    const angleStep = cardCount > 1 ? spreadDeg / (cardCount - 1) : 0;
    const startAngle = -spreadDeg / 2;

    const totalWidth = Math.min(maxOffset * (cardCount - 1), 400);
    const baseOffset = cardCount > 1 ? totalWidth / (cardCount - 1) : 0;

    sorted.forEach(({ card, origIndex }, idx) => {
        const cardEl = createCardElement(card, origIndex);
        const angle = startAngle + angleStep * idx;
        const xOffset = -totalWidth / 2 + baseOffset * idx;
        const raise = Math.abs(angle) * 0.6; // edges lower, center higher

        cardEl.style.setProperty('--card-offset', `${xOffset}px`);
        cardEl.style.setProperty('--card-rotation', `${angle}deg`);
        cardEl.style.setProperty('--card-raise', `${raise}px`);
        cardEl.style.zIndex = idx + 1;
        
        // Animate new cards drawn
        if (newIndices.has(origIndex)) {
            cardEl.classList.add('animating-draw');
            setTimeout(() => cardEl.classList.remove('animating-draw'), 500);
        }

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
        card.innerHTML = `<img src="/static/assets/cards/back/deck2_back.png" alt="Gegner Karte">`;
        card.style.transform = `translateX(${xOffset}px) translateY(${raise}px) rotate(${angle}deg)`;
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
    
    // Click: handle selection (numbers) and immediate plays
    cardEl.addEventListener('click', () => {
        if (gameState.current_turn !== 'player') {
            if (statusMessageEl) statusMessageEl.textContent = 'Nicht dein Zug!';
            return;
        }
        
        if (card.type === 'number') {
            // In sum mode, allow selection of two cards
            if (sumModeActive) {
                const i = selectedIndices.indexOf(index);
                if (i >= 0) {
                    // Deselect this card
                    selectedIndices.splice(i, 1);
                    cardEl.classList.remove('selected');
                    if (statusMessageEl) {
                        statusMessageEl.textContent = selectedIndices.length === 0 
                            ? 'Summen-Modus: Wähle zwei Karten'
                            : 'Summen-Modus: Wähle eine weitere Karte';
                    }
                    return;
                }
                
                if (selectedIndices.length >= 2) {
                    // Already have 2 cards selected, ignore
                    if (statusMessageEl) statusMessageEl.textContent = 'Zwei Karten bereits ausgewählt!';
                    return;
                }
                
                selectedIndices.push(index);
                cardEl.classList.add('selected');
                
                if (selectedIndices.length === 2) {
                    // Two cards selected, play the sum
                    const [a, b] = selectedIndices;
                    selectedIndices = [];
                    document.querySelectorAll('.hand-card.selected').forEach(n => n.classList.remove('selected'));
                    const gid = sessionStorage.getItem('gameId');
                    socket.emit('play_sum', { game_id: gid, index_a: a, index_b: b });
                    
                    // Turn off sum mode after playing
                    sumModeActive = false;
                    if (sumModeToggleBtn) sumModeToggleBtn.classList.remove('active');
                } else {
                    if (statusMessageEl) statusMessageEl.textContent = 'Summen-Modus: Wähle eine weitere Karte';
                }
            } else {
                // Normal mode: play single card immediately
                playCard(index);
            }
        } else {
            // Action card: check if it needs opponent selection or TAUSCH
            if (card.action === 'TAUSCH') {
                // Show modal to select which pile to take from
                showTauschModal(index);
            } else if (card.action === 'RESET') {
                const targetOpponent = 1;
                playCard(index, card, targetOpponent);
            } else if (card.action === 'GESCHENK') {
                showGeschenkModal(index);
            } else {
                playCard(index, card);
            }
        }
    });
    
    return cardEl;
}

// Get card image path
function getCardImagePath(card) {
    if (card.type === 'number') {
        const colorMap = {
            'ROT': 'rot',
            'BLAU': 'blau',
            'GELB': 'gelb',
            'VIOLETT': 'violett'
        };
        const colorLower = colorMap[card.color] || card.color.toLowerCase();
        return `/static/assets/cards/number/${colorLower}_${card.value}_copy1.png`;
    } else if (card.type === 'action') {
        const actionMap = {
            'AUSSETZEN': 'aussetzen',
            'TAUSCH': 'tausch',
            'PLUS2': 'plus2',
            'RICHTUNGSWECHSEL': 'richtungswechsel',
            'JOKER': 'joker',
            'GESCHENK': 'geschenk',
            'RESET': 'reset'
        };
        const actionLower = actionMap[card.action] || card.action.toLowerCase();
        return `/static/assets/cards/action/${actionLower}_1.png`;
    }
    return '/static/assets/cards/back/deck2_back.png';
}

// Render number discard pile (stack recent cards)
function renderNumberDiscardPile(cards) {
    if (!discardPileEl) return;
    discardPileEl.innerHTML = '';
    const recent = cards.slice(-4);
    recent.forEach((card) => {
        const cardEl = document.createElement('div');
        cardEl.className = 'discard-card';
        const imagePath = getCardImagePath(card);
        cardEl.innerHTML = `<img src="${imagePath}" alt="Discard">`;
        discardPileEl.appendChild(cardEl);
    });
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
function playCard(cardIndex, card = null, targetOpponent = null) {
    if (gameState.current_turn !== 'player') {
        statusMessageEl.textContent = "It's not your turn!";
        return;
    }
    
    // Find and animate the card
    const cardEl = document.querySelector(`[data-index="${cardIndex}"]`);
    if (cardEl) {
        cardEl.classList.add('animating-play');
        setTimeout(() => cardEl.classList.remove('animating-play'), 600);
    }
    
    const gid = sessionStorage.getItem('gameId');
    const payload = { game_id: gid, card_index: cardIndex };
    if (targetOpponent !== null) {
        payload.target_opponent = targetOpponent;
    }
    socket.emit('play_card', payload);
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
        const gid = sessionStorage.getItem('gameId');
        socket.emit('draw_card', { game_id: gid });
    });
}

// Handle game over
function handleGameOver(data) {
    const message = data.winner === 'player' 
        ? '🎉 Sieg! Du hast deinen CODE erreicht!' 
        : '😔 Niederlage. Der Gegner war schneller.';
    
    if (statusMessageEl) statusMessageEl.textContent = message;
    if (turnIndicatorEl) turnIndicatorEl.textContent = 'Game Over';
    
    // Reveal opponent details if they won
    if (data.winner === 'opponent') {
        if (Array.isArray(data.opponent_hand)) {
            renderOpponentReveal(data.opponent_hand);
        }
        if (Array.isArray(data.opponent_code)) {
            try {
                const codeStr = data.opponent_code.map(String).join('');
                if (statusMessageEl) statusMessageEl.textContent = `${message} | Gegner-Code: ${codeStr}`;
            } catch {}
        }
    }

    // Show play again button after delay
    setTimeout(() => {
        if (confirm(message + '\n\nNochmal spielen?')) {
            window.location.href = '/';
        }
    }, 2000);
}

function renderOpponentReveal(cards) {
    if (!opponentHandEl) return;
    opponentHandEl.innerHTML = '';
    const count = Math.min(cards.length, 12);
    const maxSpreadDeg = 22;
    const maxOffset = 38;
    const spreadDeg = count > 1 ? Math.min(maxSpreadDeg, (count - 1) * 5) : 0;
    const angleStep = count > 1 ? spreadDeg / (count - 1) : 0;
    const startAngle = -spreadDeg / 2;
    const totalWidth = Math.min(maxOffset * (count - 1), 400);
    const baseOffset = count > 1 ? totalWidth / (count - 1) : 0;
    for (let i = 0; i < count; i++) {
        const angle = startAngle + angleStep * i;
        const xOffset = -totalWidth / 2 + baseOffset * i;
        const raise = Math.abs(angle) * 0.9;
        const card = document.createElement('div');
        card.className = 'opponent-card';
        card.style.setProperty('--card-z', String(i + 1));
        card.style.transform = `translateX(${xOffset}px) translateY(${raise}px) rotate(${angle}deg)`;
        const imgPath = getCardImagePath(cards[i]);
        card.innerHTML = `<img src="${imgPath}" alt="Gegner Karte">`;
        opponentHandEl.appendChild(card);
    }
    const opponentCount = document.getElementById('opponentCount');
    if (opponentCount) opponentCount.textContent = `${cards.length}`;
}

// Request legal moves from server
function requestLegalMoves() {
    if (gameState.current_turn !== 'player') {
        legalMoves = [];
        renderLegalMoves();
        return;
    }
    const gid = sessionStorage.getItem('gameId');
    socket.emit('get_legal_moves', { game_id: gid });
}

// Render legal moves in the sidebar
function renderLegalMoves() {
    if (!movesListEl) return;
    movesListEl.innerHTML = '';
    
    if (gameState.current_turn !== 'player') {
        movesListEl.innerHTML = '<div class="move-option disabled">Warten auf Gegner...</div>';
        return;
    }
    
    if (legalMoves.length === 0) {
        movesListEl.innerHTML = '<div class="move-option disabled">Keine Züge verfügbar</div>';
        return;
    }
    
    legalMoves.forEach((move) => {
        const moveEl = document.createElement('div');
        moveEl.className = 'move-option';
        moveEl.textContent = move.description;
        moveEl.dataset.moveIndex = move.index;
        moveEl.dataset.moveType = move.type;
        
        // Store move data for execution
        if (move.card_index !== undefined) {
            moveEl.dataset.cardIndex = move.card_index;
        }
        if (move.card_indices) {
            moveEl.dataset.cardIndices = JSON.stringify(move.card_indices);
        }
        
        moveEl.addEventListener('click', () => executeMove(move));
        movesListEl.appendChild(moveEl);
    });
}

// Execute a move from the legal moves panel
function executeMove(move) {
    if (gameState.current_turn !== 'player') {
        if (statusMessageEl) statusMessageEl.textContent = 'Nicht dein Zug!';
        return;
    }
    
    const gid = sessionStorage.getItem('gameId');
    
    if (move.type === 'Draw') {
        socket.emit('draw_card', { game_id: gid });
    } else if (move.type === 'PlaySum') {
        const [a, b] = move.card_indices;
        socket.emit('play_sum', { game_id: gid, index_a: a, index_b: b });
    } else if (move.type === 'PlayAction' || move.type === 'PlayNumber' || move.type === 'PlayAny') {
        // For RESET and GESCHENK, target_opponent is always 1 in 2-player game
        const payload = { game_id: gid, card_index: move.card_index };
        
        // Check if this is a card that needs opponent selection
        const card = gameState.player_hand[move.card_index];
        if (card && card.type === 'action') {
            if (card.action === 'GESCHENK') {
                showGeschenkModal(move.card_index);
                return;
            }
            if (card.action === 'RESET') {
                payload.target_opponent = 1;
            }
        }
        
        socket.emit('play_card', payload);
    }
}

// TAUSCH modal functions
let pendingTauschCardIndex = null;
let pendingGeschenkCardIndex = null;

function showTauschModal(cardIndex) {
    pendingTauschCardIndex = cardIndex;
    const modal = document.getElementById('tauschModal');
    const cardsGrid = document.getElementById('tauschCardsGrid');
    const actionOption = document.getElementById('tauschActionOption');
    const actionPreview = document.getElementById('tauschActionPreview');
    
    // Clear previous cards
    cardsGrid.innerHTML = '';
    
    // Handle top action card option
    const hasActionCards = Array.isArray(gameState.action_discard) && gameState.action_discard.length > 0;
    if (actionOption) {
        actionOption.classList.toggle('disabled', !hasActionCards);
        if (actionPreview) {
            actionPreview.textContent = hasActionCards
                ? 'Oberste Karte vom Aktionsstapel'
                : 'Keine Aktionskarten verfügbar';
        }
    }
    
    // Populate with action discard pile cards
    if (gameState.action_discard && gameState.action_discard.length > 0) {
        gameState.action_discard.forEach((card, idx) => {
            const cardEl = document.createElement('div');
            cardEl.className = 'tausch-card-option';
            cardEl.innerHTML = `<span class="card-action-symbol">${card.action}</span>`;
            cardEl.addEventListener('click', () => selectTauschTarget('action', idx));
            cardsGrid.appendChild(cardEl);
        });
    } else {
        cardsGrid.innerHTML = '<p style="color: #999; text-align: center;">Keine Aktionskarten abgelegt</p>';
    }
    
    modal.classList.add('active');
}

function selectTauschTarget(target, actionIndex = null) {
    const card = gameState.player_hand[pendingTauschCardIndex];
    const gid = sessionStorage.getItem('gameId');
    
    const payload = {
        game_id: gid,
        card_index: pendingTauschCardIndex,
        target: target  // 'number' or 'action'
    };
    
    if (target === 'action' && actionIndex !== null) {
        payload.action_index = actionIndex;
    }
    
    socket.emit('play_card', payload);
    closeTauschModal();
}

function closeTauschModal() {
    const modal = document.getElementById('tauschModal');
    modal.classList.remove('active');
    pendingTauschCardIndex = null;
}

// GESCHENK modal functions
function showGeschenkModal(cardIndex) {
    pendingGeschenkCardIndex = cardIndex;
    requestOpponentHand();
    populateGeschenkModal();
    const modal = document.getElementById('geschenkModal');
    if (modal) modal.classList.add('active');
}

function populateGeschenkModal() {
    const grid = document.getElementById('geschenkCardsGrid');
    if (!grid) return;
    grid.innerHTML = '';
    if (!opponentRevealHand || opponentRevealHand.length === 0) {
        grid.innerHTML = '<p style="color:#999; text-align:center;">Keine Karten sichtbar</p>';
        return;
    }
    opponentRevealHand.forEach((card, idx) => {
        const cardEl = document.createElement('div');
        cardEl.className = 'tausch-card-option';
        const imagePath = getCardImagePath(card);
        cardEl.innerHTML = `<img src="${imagePath}" alt="${card.action || card.value}">`;
        cardEl.addEventListener('click', () => selectGeschenkTarget(idx));
        grid.appendChild(cardEl);
    });
}

function selectGeschenkTarget(cardIdx) {
    const gid = sessionStorage.getItem('gameId');
    const payload = {
        game_id: gid,
        card_index: pendingGeschenkCardIndex,
        target_opponent: 1,
        gift_index: cardIdx
    };
    socket.emit('play_card', payload);
    closeGeschenkModal();
}

function closeGeschenkModal() {
    const modal = document.getElementById('geschenkModal');
    if (modal) modal.classList.remove('active');
    pendingGeschenkCardIndex = null;
}

// Attach close button handler
document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('tauschModal');
    const closeBtn = document.querySelector('.tausch-modal-close');
    const actionOption = document.getElementById('tauschActionOption');
    const geschenkModal = document.getElementById('geschenkModal');
    const geschenkClose = document.getElementById('geschenkModalClose');
    
    if (closeBtn) {
        closeBtn.addEventListener('click', closeTauschModal);
    }
    if (geschenkClose) {
        geschenkClose.addEventListener('click', closeGeschenkModal);
    }
    
    // Close modal when clicking outside
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeTauschModal();
            }
        });
    }
    if (geschenkModal) {
        geschenkModal.addEventListener('click', (e) => {
            if (e.target === geschenkModal) {
                closeGeschenkModal();
            }
        });
    }
    
    const numberOption = document.querySelector('.tausch-option');
    if (numberOption) {
        numberOption.addEventListener('click', () => selectTauschTarget('number'));
    }
    if (actionOption) {
        actionOption.addEventListener('click', () => selectTauschTarget('action'));
    }
});
