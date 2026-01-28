// Main menu JavaScript
const socket = io();

// DOM Elements
const playBtn = document.getElementById('playBtn');
const setupModal = document.getElementById('setupModal');
const cancelBtn = document.getElementById('cancelBtn');
const startGameBtn = document.getElementById('startGameBtn');
const playerNameInput = document.getElementById('playerName');
const navDots = document.querySelectorAll('.nav-dot');
const agentRadios = document.querySelectorAll('input[name="agent"]');

// Show setup modal
playBtn.addEventListener('click', () => {
    setupModal.classList.add('show');
    playerNameInput.focus();
});

// Hide setup modal
cancelBtn.addEventListener('click', () => {
    setupModal.classList.remove('show');
});

// Navigation dots - go back to start page
navDots.forEach(dot => {
    dot.addEventListener('click', () => {
        setupModal.classList.remove('show');
    });
});

// Agent selection - update button color based on selection
agentRadios.forEach(radio => {
    radio.addEventListener('change', () => {
        // Update button color based on selected agent
        const selectedAgent = document.querySelector('input[name="agent"]:checked').value;
        if (selectedAgent === 'rule_based') {
            startGameBtn.classList.add('red-agent');
        } else {
            startGameBtn.classList.remove('red-agent');
        }
    });
});

// Initialize button color
const initialAgent = document.querySelector('input[name="agent"]:checked').value;
if (initialAgent === 'rule_based') {
    startGameBtn.classList.add('red-agent');
}

// Start game
startGameBtn.addEventListener('click', () => {
    const playerName = playerNameInput.value.trim();
    
    // Validate that name is not empty
    if (!playerName) {
        playerNameInput.classList.add('error');
        playerNameInput.placeholder = 'Name erforderlich!';
        return;
    }
    
    playerNameInput.classList.remove('error');
    const agentType = document.querySelector('input[name="agent"]:checked').value;
    const trainingMode = document.getElementById('trainingModeCheckbox').checked;
    
    // Store player info in sessionStorage
    sessionStorage.setItem('playerName', playerName);
    sessionStorage.setItem('agentType', agentType);
    
    // Emit start game event
    socket.emit('start_game', {
        player_name: playerName,
        agent_type: agentType,
        training_mode: trainingMode
    });
});

// Listen for game started event
socket.on('game_started', (data) => {
    console.log('Game started:', data);
    // Persist game_id for session rejoin on /game
    if (data && data.game_id) {
        sessionStorage.setItem('gameId', data.game_id);
    }
    // Redirect to game page
    window.location.href = '/game';
});

// Socket connection events
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
});

// Enable Enter key to start game
playerNameInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        startGameBtn.click();
    }
});
// Clear error state when user types
playerNameInput.addEventListener('input', () => {
    if (playerNameInput.classList.contains('error')) {
        playerNameInput.classList.remove('error');
        playerNameInput.placeholder = 'Name eingeben';
    }
});