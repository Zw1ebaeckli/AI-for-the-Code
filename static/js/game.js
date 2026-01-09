// Game board JavaScript
const socket = io();

console.log('Socket.io initialized');

socket.on('connect', () => {
    console.log('Socket connected, id:', socket.id);
});

socket.on('disconnect', () => {
    console.log('Socket disconnected');
});

// Catch all events for debugging
socket.onAny((event, ...args) => {
    console.log('Socket event received:', event, args);
});

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

// Code card drag functionality
let isDraggingCodeCard = false;
let dragOffsetX = 0;
let dragOffsetY = 0;
const codeWindowBodyEl = document.querySelector('.code-window-body');

if (playerCodeCardEl && codeWindowBodyEl) {
    playerCodeCardEl.addEventListener('mousedown', (e) => {
        isDraggingCodeCard = true;
        dragOffsetX = e.clientX - playerCodeCardEl.getBoundingClientRect().left;
        dragOffsetY = e.clientY - playerCodeCardEl.getBoundingClientRect().top;
        playerCodeCardEl.classList.add('dragging');
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDraggingCodeCard || !codeWindowBodyEl) return;

        const bodyRect = codeWindowBodyEl.getBoundingClientRect();
        let newX = e.clientX - bodyRect.left - dragOffsetX;
        let newY = e.clientY - bodyRect.top - dragOffsetY;

        // Constrain card within the code window body
        const cardWidth = playerCodeCardEl.offsetWidth;
        const cardHeight = playerCodeCardEl.offsetHeight;
        const padding = 10;

        newX = Math.max(padding, Math.min(newX, bodyRect.width - cardWidth - padding));
        newY = Math.max(padding, Math.min(newY, bodyRect.height - cardHeight - padding));

        playerCodeCardEl.style.position = 'absolute';
        playerCodeCardEl.style.left = newX + 'px';
        playerCodeCardEl.style.top = newY + 'px';
        playerCodeCardEl.style.margin = '0';
    });

    document.addEventListener('mouseup', () => {
        if (isDraggingCodeCard) {
            isDraggingCodeCard = false;
            playerCodeCardEl.classList.remove('dragging');
        }
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
let playableCardIndices = new Set();
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

// Game Over button handlers
const btnPlayAgain = document.getElementById('btnPlayAgain');
const btnReturnMenu = document.getElementById('btnReturnMenu');

if (btnPlayAgain) {
    btnPlayAgain.addEventListener('click', () => {
        // Clear current game/session but keep player settings
        const name = sessionStorage.getItem('playerName');
        const agent = sessionStorage.getItem('agentType');
        sessionStorage.clear();
        if (name) sessionStorage.setItem('playerName', name);
        if (agent) sessionStorage.setItem('agentType', agent);
        window.location.href = '/game';
    });
}

if (btnReturnMenu) {
    btnReturnMenu.addEventListener('click', () => {
        // Return to main menu and clear session
        sessionStorage.clear();
        window.location.href = '/';
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
    // Build set of playable single-card indices for hover/click feedback
    const playable = new Set();
    for (const m of legalMoves) {
        if (m.type === 'PlayNumber' || m.type === 'PlayAny' || m.type === 'PlayAction') {
            if (typeof m.card_index === 'number') playable.add(m.card_index);
        }
    }
    playableCardIndices = playable;
    updatePlayableClasses();
    renderLegalMoves();
});

socket.on('card_drawn', (data) => {
    console.log('\n=== CARD DRAWN EVENT RECEIVED ===');
    console.log('Card:', data.card);
    console.log('Is playable:', data.is_playable);
    console.log('Available moves:', data.available_moves);
    console.log('=================================\n');
    
    // Only show modal if card is playable
    if (data.is_playable) {
        showDrawnCardModal(data.card, data.available_moves, data.is_playable);
    } else {
        console.log('Card not playable, not showing modal');
    }
});

// Animate opponent steps and update state progressively
socket.on('opponent_move', (data) => {
    try {
        const state = data.state;
        const moveDescription = data.move_description || 'Gegner hat eine Aktion durchgeführt';
        
        // Handle GESCHENK animation if present
        if (data.geschenk_data && data.geschenk_data.taken_card) {
            animateGeschenkEffect(data.geschenk_data.taken_card);
        }
        
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
    
    // Handle +2 penalty state
    if (state.pending_plus2 && state.pending_plus2 > 0 && state.current_turn === 'player') {
        handlePendingPlus2(state.player_hand);
    } else {
        clearPlus2State();
    }
}

// Handle +2 penalty - grey out all cards except +2, highlight draw pile if no +2
function handlePendingPlus2(playerHand) {
    const hasPlus2 = playerHand.some(card => card.type === 'action' && card.action === 'PLUS2');
    
    // Grey out all cards except +2
    document.querySelectorAll('.hand-card').forEach((cardEl) => {
        const idx = parseInt(cardEl.dataset.index, 10);
        const card = playerHand[idx];
        
        if (card && card.type === 'action' && card.action === 'PLUS2') {
            // Keep +2 cards enabled
            cardEl.style.filter = 'none';
            cardEl.style.opacity = '1';
            cardEl.classList.remove('unplayable');
        } else {
            // Grey out other cards
            cardEl.style.filter = 'grayscale(100%) brightness(0.75) contrast(0.9)';
            cardEl.style.opacity = '1';
            cardEl.classList.add('unplayable');
        }
    });
    
    // Add blue border to draw pile if no +2 cards available
    const drawPileEl = document.getElementById('drawPile');
    if (drawPileEl) {
        if (!hasPlus2) {
            drawPileEl.style.border = '3px solid #2196F3';
            drawPileEl.style.borderRadius = '12px';
            drawPileEl.style.boxShadow = '0 0 15px rgba(33, 150, 243, 0.6)';
        } else {
            drawPileEl.style.border = '';
            drawPileEl.style.boxShadow = '';
        }
    }
}

// Clear +2 penalty state styling
function clearPlus2State() {
    // Remove blue border from draw pile
    const drawPileEl = document.getElementById('drawPile');
    if (drawPileEl) {
        drawPileEl.style.border = '';
        drawPileEl.style.boxShadow = '';
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
        if (!playableCardIndices.has(origIndex)) {
            cardEl.classList.add('unplayable');
        }
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

        // If not playable as a single move and not in sum mode, shake + red overlay
        if (!sumModeActive && !playableCardIndices.has(index)) {
            cardEl.classList.add('invalid', 'shake');
            setTimeout(() => cardEl.classList.remove('shake'), 500);
            setTimeout(() => cardEl.classList.remove('invalid'), 700);
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
                    if (sumModeToggleBtn) {
                        sumModeToggleBtn.classList.remove('active');
                        // Unlock the sum toggle after sum is played
                        sumModeToggleBtn.disabled = false;
                        sumModeToggleBtn.style.pointerEvents = 'auto';
                        sumModeToggleBtn.style.opacity = '1';
                    }
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

// Update CSS classes for playable/unplayable cards without re-rendering
function updatePlayableClasses() {
    document.querySelectorAll('.hand-card').forEach((el) => {
        const idx = parseInt(el.dataset.index, 10);
        if (Number.isFinite(idx)) {
            if (playableCardIndices.has(idx)) {
                el.classList.remove('unplayable');
            } else {
                el.classList.add('unplayable');
            }
        }
    });
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
    discardPileEl.classList.toggle('has-stack', Array.isArray(cards) && cards.length > 0);
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
    const isWin = data.winner === 'player';
    
    // Show game over overlay
    showGameOverScreen(isWin);
    
    // Keep status message updated
    const message = isWin 
        ? 'Sieg! Du hast deinen CODE erreicht!' 
        : 'Niederlage. Der Gegner war schneller.';
    
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
}

function showGameOverScreen(isWin) {
    const overlay = document.getElementById('gameOverOverlay');
    const title = document.getElementById('gameOverTitle');
    const confettiContainer = document.getElementById('confettiContainer');
    
    if (!overlay || !title) return;
    
    // Set title and styling
    if (isWin) {
        title.textContent = 'Du hast gewonnen!';
        title.className = 'game-over-title won';
        createConfetti(confettiContainer);
    } else {
        title.textContent = 'Du wurdest besiegt!';
        title.className = 'game-over-title lost';
        confettiContainer.innerHTML = ''; // No confetti for loss
    }
    
    // Show overlay
    overlay.classList.add('active');
}

function createConfetti(container) {
    if (!container) return;
    
    container.innerHTML = ''; // Clear existing
    
    // Create 100 confetti pieces
    for (let i = 0; i < 100; i++) {
        const confetti = document.createElement('div');
        confetti.className = 'confetti';
        
        // Random horizontal position
        confetti.style.left = Math.random() * 100 + '%';
        
        // Random animation delay
        confetti.style.animationDelay = Math.random() * 3 + 's';
        
        // Random animation duration
        confetti.style.animationDuration = (2 + Math.random() * 2) + 's';
        
        // Random size
        const size = 8 + Math.random() * 8;
        confetti.style.width = size + 'px';
        confetti.style.height = size + 'px';
        
        // Random rotation
        confetti.style.transform = `rotate(${Math.random() * 360}deg)`;
        
        container.appendChild(confetti);
    }
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
    const numberOption = document.querySelector('.tausch-option[data-target="number"], .tausch-option');
    const actionPreview = document.getElementById('tauschActionPreview');
    const numberPreviewImg = document.getElementById('tauschNumberPreviewImg');
    const actionPreviewImg = document.getElementById('tauschActionPreviewImg');
    
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
        if (actionPreviewImg) {
            if (hasActionCards) {
                const topAction = gameState.action_discard[gameState.action_discard.length - 1];
                actionPreviewImg.src = getCardImagePath(topAction);
                actionPreviewImg.style.display = 'block';
            } else {
                actionPreviewImg.style.display = 'none';
            }
        }
    }

    // Handle top number card preview and availability (must have >1 card to take)
    const hasNumber = Array.isArray(gameState.number_discard) && gameState.number_discard.length > 1;
    if (numberPreviewImg) {
        if (hasNumber) {
            const topNum = gameState.number_discard[gameState.number_discard.length - 1];
            numberPreviewImg.src = getCardImagePath(topNum);
            numberPreviewImg.style.display = 'block';
        } else {
            numberPreviewImg.style.display = 'none';
        }
    }
    if (numberOption) {
        numberOption.classList.toggle('disabled', !hasNumber);
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
    // Prevent picking number discard if only one card is present
    if (target === 'number') {
        const hasNumber = Array.isArray(gameState.number_discard) && gameState.number_discard.length > 1;
        if (!hasNumber) {
            if (statusMessageEl) statusMessageEl.textContent = 'Keine Zahlentausch-Karte verfügbar (nur eine Karte im Stapel).';
            return;
        }
    }
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
    if (statusMessageEl) statusMessageEl.textContent = 'Geschenk wird gespielt...';
    console.log('GESCHENK select index', cardIdx);
    console.log('Pending Geschenk card index:', pendingGeschenkCardIndex);
    const payload = {
        game_id: gid,
        card_index: pendingGeschenkCardIndex,
        target_opponent: 1,
        gift_index: cardIdx
    };
    console.log('Sending GESCHENK payload:', payload);
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
    
    const numberOption = document.querySelector('.tausch-option[data-target="number"], .tausch-option');
    if (numberOption) {
        numberOption.addEventListener('click', () => selectTauschTarget('number'));
    }
    if (actionOption) {
        actionOption.addEventListener('click', () => selectTauschTarget('action'));
    }
    
    // Drawn card modal close button
    const drawnCardModal = document.getElementById('drawnCardModal');
    const drawnCardCloseBtn = document.getElementById('drawnCardModalClose');
    const drawnCardAddToHandBtn = document.getElementById('drawnCardAddToHandBtn');
    
    if (drawnCardCloseBtn) {
        drawnCardCloseBtn.addEventListener('click', () => {
            // Closing the modal without playing = add to hand
            const gid = sessionStorage.getItem('gameId');
            socket.emit('drawn_card_add_to_hand', { game_id: gid });
            closeDrawnCardModal();
        });
    }
    if (drawnCardAddToHandBtn) {
        drawnCardAddToHandBtn.addEventListener('click', () => {
            const gid = sessionStorage.getItem('gameId');
            socket.emit('drawn_card_add_to_hand', { game_id: gid });
            closeDrawnCardModal();
        });
    }
    if (drawnCardModal) {
        drawnCardModal.addEventListener('click', (e) => {
            if (e.target === drawnCardModal) {
                // Clicking outside = add to hand
                const gid = sessionStorage.getItem('gameId');
                socket.emit('drawn_card_add_to_hand', { game_id: gid });
                closeDrawnCardModal();
            }
        });
    }
    
    // Code window collapse/expand controls
    const codeWindowEl = document.getElementById('codeCardWindow');
    const codeWindowDotsEl = document.getElementById('codeCardToggle');
    console.log('Code window elements:', { codeWindowEl, codeWindowDotsEl });
    if (codeWindowDotsEl && codeWindowEl) {
        codeWindowDotsEl.title = 'Minimieren/Erweitern';
        codeWindowDotsEl.addEventListener('click', (e) => {
            console.log('Code window toggle clicked');
            codeWindowEl.classList.toggle('collapsed');
            console.log('Code window classList:', codeWindowEl.classList.toString());
            e.stopPropagation();
        });
    }

    // Mechanics window collapse/expand controls
    const mechanicsWindowEl = document.getElementById('mechanicsWindow');
    const mechanicsDotsEl = document.getElementById('mechanicsToggle');
    console.log('Mechanics window elements:', { mechanicsWindowEl, mechanicsDotsEl });
    if (mechanicsDotsEl && mechanicsWindowEl) {
        mechanicsDotsEl.title = 'Minimieren/Erweitern';
        mechanicsDotsEl.addEventListener('click', (e) => {
            console.log('Mechanics window toggle clicked');
            mechanicsWindowEl.classList.toggle('collapsed');
            console.log('Mechanics window classList:', mechanicsWindowEl.classList.toString());
            e.stopPropagation();
        });
    }
});

// Track the drawn card for sum mode handling
let currentDrawnCard = null;

function showDrawnCardModal(drawnCard, availableMoves, isPlayable) {
    console.log('\\n=== SHOWING DRAWN CARD MODAL ===');
    console.log('Card:', drawnCard);
    console.log('Is playable:', isPlayable);
    console.log('Available moves:', availableMoves);
    
    currentDrawnCard = drawnCard; // Store for later reference
    
    const modal = document.getElementById('drawnCardModal');
    const displayEl = document.getElementById('drawnCardDisplay');
    const optionsEl = document.getElementById('drawnCardPlayOptions');
    const addToHandBtn = document.getElementById('drawnCardAddToHandBtn');
    
    if (!modal || !displayEl || !optionsEl) {
        console.error('Modal elements not found!', {modal, displayEl, optionsEl});
        return;
    }
    
    // Show the drawn card image
    const imagePath = getCardImagePath(drawnCard);
    displayEl.innerHTML = `<img src="${imagePath}" alt="Gezogene Karte" style="width: 120px; height: auto; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">`;
    
    // Clear previous options
    optionsEl.innerHTML = '';
    
    if (isPlayable && availableMoves && availableMoves.length > 0) {
        // Card is playable - show play options
        console.log('Card IS playable - creating buttons');
        
        // Group moves by type for display
        const moveGroups = {};
        availableMoves.forEach(move => {
            if (!moveGroups[move.move_type]) {
                moveGroups[move.move_type] = [];
            }
            moveGroups[move.move_type].push(move);
        });
        
        console.log('Move groups:', moveGroups);
        
        // Create buttons for each move type
        if (moveGroups['PlayNumber']) {
            const btn = document.createElement('button');
            btn.className = 'action-button';
            btn.style.cssText = `
                width: 100%;
                padding: 0.75rem 1rem;
                margin-bottom: 0.5rem;
                background: linear-gradient(135deg, #4CAF50, #45a049);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: bold;
                cursor: pointer;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                transition: all 0.2s;
            `;
            btn.textContent = '▶ Karte direkt spielen';
            btn.onmouseover = () => btn.style.transform = 'translateY(-2px)';
            btn.onmouseout = () => btn.style.transform = 'translateY(0)';
            btn.addEventListener('click', () => playDrawnCardImmediately('PlayNumber', drawnCard));
            optionsEl.appendChild(btn);
        }
        
        if (moveGroups['PlaySum']) {
            const btn = document.createElement('button');
            btn.className = 'action-button';
            btn.style.cssText = `
                width: 100%;
                padding: 0.75rem 1rem;
                margin-bottom: 0.5rem;
                background: linear-gradient(135deg, #2196F3, #1976D2);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: bold;
                cursor: pointer;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                transition: all 0.2s;
            `;
            btn.textContent = 'Als Summe spielen';
            btn.onmouseover = () => btn.style.transform = 'translateY(-2px)';
            btn.onmouseout = () => btn.style.transform = 'translateY(0)';
            btn.addEventListener('click', () => activateSumModeForDrawnCard(drawnCard));
            optionsEl.appendChild(btn);
        }
        
        if (moveGroups['PlayAction']) {
            const btn = document.createElement('button');
            btn.className = 'action-button';
            btn.style.cssText = `
                width: 100%;
                padding: 0.75rem 1rem;
                margin-bottom: 0.5rem;
                background: linear-gradient(135deg, #FF9800, #F57C00);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: bold;
                cursor: pointer;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                transition: all 0.2s;
            `;
            btn.textContent = `${drawnCard.action} spielen`;
            btn.onmouseover = () => btn.style.transform = 'translateY(-2px)';
            btn.onmouseout = () => btn.style.transform = 'translateY(0)';
            btn.addEventListener('click', () => playDrawnCardImmediately('PlayAction', drawnCard));
            optionsEl.appendChild(btn);
        }
        
        if (addToHandBtn) {
            addToHandBtn.textContent = 'Zur Hand hinzufügen (nicht spielen)';
            addToHandBtn.style.cssText = `
                width: 100%;
                padding: 0.9rem 1rem;
                margin-top: 0.75rem;
                background: linear-gradient(180deg, #ECEFF1 0%, #CFD8DC 100%);
                color: #263238;
                border: 1px solid #B0BEC5;
                border-radius: 12px;
                font-size: 0.98rem;
                font-weight: 600;
                letter-spacing: 0.2px;
                cursor: pointer;
                box-shadow: 0 2px 6px rgba(0,0,0,0.12), inset 0 1px 0 rgba(255,255,255,0.6);
                transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
            `;
            addToHandBtn.onmouseover = () => {
                addToHandBtn.style.transform = 'translateY(-1px)';
                addToHandBtn.style.boxShadow = '0 4px 10px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.65)';
                addToHandBtn.style.background = 'linear-gradient(180deg, #F1F5F9 0%, #DDE3E8 100%)';
            };
            addToHandBtn.onmouseout = () => {
                addToHandBtn.style.transform = 'translateY(0)';
                addToHandBtn.style.boxShadow = '0 2px 6px rgba(0,0,0,0.12), inset 0 1px 0 rgba(255,255,255,0.6)';
                addToHandBtn.style.background = 'linear-gradient(180deg, #ECEFF1 0%, #CFD8DC 100%)';
            };
        }
    } else {
        // Card is NOT playable - show message
        console.log('Card is NOT playable - showing message');
        const msg = document.createElement('div');
        msg.className = 'not-playable-message';
        msg.textContent = 'Diese Karte kann nicht gespielt werden';
        msg.style.cssText = 'color: #ff6b6b; padding: 1rem; text-align: center; font-weight: bold; font-size: 1.1rem;';
        optionsEl.appendChild(msg);
        
        if (addToHandBtn) {
            addToHandBtn.textContent = 'Zur Hand hinzufügen (nicht spielen)';
            // Ensure consistent secondary styling even in non-playable path
            addToHandBtn.style.cssText = `
                width: 100%;
                padding: 0.9rem 1rem;
                margin-top: 0.75rem;
                background: linear-gradient(180deg, #ECEFF1 0%, #CFD8DC 100%);
                color: #263238;
                border: 1px solid #B0BEC5;
                border-radius: 12px;
                font-size: 0.98rem;
                font-weight: 600;
                letter-spacing: 0.2px;
                cursor: pointer;
                box-shadow: 0 2px 6px rgba(0,0,0,0.12), inset 0 1px 0 rgba(255,255,255,0.6);
                transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
            `;
        }
    }
    
    console.log('Adding active class to modal');
    modal.classList.add('active');
    console.log('=================================\\n');
}

function closeDrawnCardModal() {
    const modal = document.getElementById('drawnCardModal');
    if (modal) modal.classList.remove('active');
}

function activateSumModeForDrawnCard(drawnCard) {
    console.log('Activating sum mode for drawn card:', drawnCard);
    
    // Close the modal
    closeDrawnCardModal();
    
    // Activate sum mode
    sumModeActive = true;
    if (sumModeToggleBtn) {
        sumModeToggleBtn.classList.add('active');
        // Lock the sum toggle while sum mode is active (from modal choice)
        sumModeToggleBtn.disabled = true;
        sumModeToggleBtn.style.pointerEvents = 'none';
        sumModeToggleBtn.style.opacity = '0.6';
    }
    
    // Find the drawn card in the player's hand and select it
    const handCards = gameState.player_hand || [];
    let drawnCardIndex = -1;
    
    // Find the last card in hand (most recently drawn)
    for (let i = handCards.length - 1; i >= 0; i--) {
        const card = handCards[i];
        if (card.type === drawnCard.type && 
            card.value === drawnCard.value && 
            (drawnCard.color ? card.color === drawnCard.color : true) &&
            (drawnCard.action ? card.action === drawnCard.action : true)) {
            drawnCardIndex = i;
            break;
        }
    }
    
    console.log('Found drawn card at index:', drawnCardIndex);
    
    if (drawnCardIndex >= 0) {
        // Mark the drawn card as selected
        selectedIndices = [drawnCardIndex];
        
        // Find what the second number needs to be (handle numeric strings)
        const topCardValueNum = Number(gameState.discard_top?.value);
        const drawnCardValueNum = Number(drawnCard.value);
        
        if (!Number.isNaN(topCardValueNum) && !Number.isNaN(drawnCardValueNum)) {
            const neededValue = topCardValueNum - drawnCardValueNum;
            console.log(`Need a card with value ${neededValue} to sum to ${topCardValueNum}`);
            
            // Grey out all cards except the drawn card and cards with the needed value
            document.querySelectorAll('.hand-card').forEach((cardEl) => {
                const idx = parseInt(cardEl.dataset.index, 10);
                
                if (idx === drawnCardIndex) {
                    // This is the drawn card - mark as selected
                    cardEl.classList.add('selected');
                    cardEl.style.filter = 'none';
                    cardEl.style.opacity = '1';
                    cardEl.style.pointerEvents = 'auto';
                } else {
                    const card = handCards[idx];
                    if (card && card.type === 'number' && Number(card.value) === neededValue) {
                        // This card can be the second digit - keep enabled
                        cardEl.style.filter = 'none';
                        cardEl.style.opacity = '1';
                        cardEl.style.pointerEvents = 'auto';
                    } else {
                        // Grey out this card without transparency
                        cardEl.style.filter = 'grayscale(100%) brightness(0.75) contrast(0.9)';
                        cardEl.style.opacity = '1';
                        cardEl.style.pointerEvents = 'none';
                    }
                }
            });
        }
    }
    
    // Update status message
    if (statusMessageEl) {
        statusMessageEl.textContent = 'Summen-Modus: Wähle eine weitere Karte zum Hinzufügen';
    }
}

function playDrawnCardImmediately(moveType, card) {
    const gid = sessionStorage.getItem('gameId');
    
    // Special handling for action cards that need additional selection
    if (moveType === 'PlayAction') {
        if (card.action === 'GESCHENK') {
            // Close modal and find the card index in hand, then show GESCHENK modal
            closeDrawnCardModal();
            // Find the card in the player's hand
            const playerHand = gameState.player_hand || [];
            const cardIndex = playerHand.findIndex(c => 
                c.type === card.type && 
                c.action === card.action && 
                JSON.stringify(c) === JSON.stringify(card)
            );
            if (cardIndex !== -1) {
                showGeschenkModal(cardIndex);
            }
            return;
        } else if (card.action === 'TAUSCH') {
            // Close modal and find the card index in hand, then show TAUSCH modal
            closeDrawnCardModal();
            // Find the card in the player's hand
            const playerHand = gameState.player_hand || [];
            const cardIndex = playerHand.findIndex(c => 
                c.type === card.type && 
                c.action === card.action && 
                JSON.stringify(c) === JSON.stringify(card)
            );
            if (cardIndex !== -1) {
                showTauschModal(cardIndex);
            }
            return;
        }
    }
    
    // For all other cards, play normally
    socket.emit('play_drawn_card', {
        game_id: gid,
        move_type: moveType,
        card: card
    });
    closeDrawnCardModal();
}

function showSumSelectionModal(drawnCard) {
    // Get current player hand from the game state
    const handCards = document.querySelectorAll('.player-card');
    const handList = [];
    handCards.forEach(cardEl => {
        const idx = Array.from(cardEl.parentElement.children).indexOf(cardEl);
        if (idx >= 0) {
            handList.push({
                index: idx,
                element: cardEl
            });
        }
    });
    
    // Simple approach: just advance to the legal moves selection
    // The player will click on a card to form a sum
    closeDrawnCardModal();
    showStatusMessage('Wähle eine Karte aus deiner Hand, um eine Summe zu bilden', 'info');
}

// Animate GESCHENK effect when opponent takes a card
function animateGeschenkEffect(takenCard) {
    console.log('Animating GESCHENK effect for card:', takenCard);
    
    // Find the card in player's hand that matches
    const playerCards = document.querySelectorAll('.hand-card');
    let cardToAnimate = null;
    
    playerCards.forEach(cardEl => {
        const img = cardEl.querySelector('img');
        if (img && img.src.includes(getCardImagePath(takenCard).split('/').pop())) {
            cardToAnimate = cardEl;
        }
    });
    
    if (!cardToAnimate) {
        console.log('Could not find card to animate');
        return;
    }
    
    // Get positions
    const cardRect = cardToAnimate.getBoundingClientRect();
    const opponentHandEl = document.querySelector('.opponent-hand');
    const drawPileEl = document.getElementById('drawPile');
    
    if (!opponentHandEl || !drawPileEl) return;
    
    const opponentRect = opponentHandEl.getBoundingClientRect();
    const drawPileRect = drawPileEl.getBoundingClientRect();
    
    // Create flying card element (player card to opponent)
    const flyingCard = document.createElement('div');
    flyingCard.className = 'flying-card';
    flyingCard.innerHTML = `<img src="${getCardImagePath(takenCard)}" alt="Card">`;
    flyingCard.style.position = 'fixed';
    flyingCard.style.left = `${cardRect.left}px`;
    flyingCard.style.top = `${cardRect.top}px`;
    flyingCard.style.width = `${cardRect.width}px`;
    flyingCard.style.height = `${cardRect.height}px`;
    flyingCard.style.zIndex = '1000';
    flyingCard.style.pointerEvents = 'none';
    flyingCard.style.transition = 'all 0.6s cubic-bezier(0.4, 0.0, 0.2, 1)';
    
    document.body.appendChild(flyingCard);
    
    // Hide original card
    cardToAnimate.style.opacity = '0';
    
    // Animate to opponent hand
    setTimeout(() => {
        flyingCard.style.left = `${opponentRect.left + opponentRect.width / 2 - cardRect.width / 2}px`;
        flyingCard.style.top = `${opponentRect.top + opponentRect.height / 2 - cardRect.height / 2}px`;
        flyingCard.style.transform = 'scale(0.7) rotate(5deg)';
        flyingCard.style.opacity = '0.8';
    }, 50);
    
    // After first animation, create draw pile to player animation
    setTimeout(() => {
        flyingCard.remove();
        
        // Create second flying card (draw pile to player)
        const drawCard = document.createElement('div');
        drawCard.className = 'flying-card';
        drawCard.innerHTML = `<img src="/static/assets/cards/back/deck2_back.png" alt="Card">`;
        drawCard.style.position = 'fixed';
        drawCard.style.left = `${drawPileRect.left}px`;
        drawCard.style.top = `${drawPileRect.top}px`;
        drawCard.style.width = `${drawPileRect.width}px`;
        drawCard.style.height = `${drawPileRect.height}px`;
        drawCard.style.zIndex = '-1';  // Lower z-index so it goes behind player cards
        drawCard.style.pointerEvents = 'none';
        drawCard.style.transition = 'all 0.6s cubic-bezier(0.4, 0.0, 0.2, 1)';
        
        document.body.appendChild(drawCard);
        
        // Animate to player hand area
        const playerHandRect = playerHandEl.getBoundingClientRect();
        setTimeout(() => {
            drawCard.style.left = `${playerHandRect.left + playerHandRect.width / 2 - drawPileRect.width / 2}px`;
            drawCard.style.top = `${playerHandRect.top + playerHandRect.height / 2 - drawPileRect.height / 2}px`;
            drawCard.style.transform = 'scale(1.1)';
        }, 50);
        
        // Clean up
        setTimeout(() => {
            drawCard.remove();
            cardToAnimate.style.opacity = '1';
        }, 650);
    }, 650);
}
