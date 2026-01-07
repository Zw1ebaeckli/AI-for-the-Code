// Matrix background generator for game screen
function generateMatrixBackground() {
    const topBg = document.querySelector('.matrix-background-top');
    const bottomBg = document.querySelector('.matrix-background-bottom');
    
    if (!topBg || !bottomBg) return;
    
    const cellWidth = 28; // px
    const cellHeight = 28; // px
    
    function fillMatrix(element) {
        const width = window.innerWidth;
        const height = 150; // height of the background strip
        
        const cols = Math.ceil(width / cellWidth);
        const rows = Math.ceil(height / cellHeight);
        
        element.style.gridTemplateColumns = `repeat(${cols}, ${cellWidth}px)`;
        element.style.gridTemplateRows = `repeat(${rows}, ${cellHeight}px)`;
        
        element.innerHTML = '';
        
        for (let i = 0; i < rows * cols; i++) {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            cell.textContent = Math.floor(Math.random() * 10);
            element.appendChild(cell);
        }
    }
    
    fillMatrix(topBg);
    fillMatrix(bottomBg);
    
    // Regenerate on window resize
    window.addEventListener('resize', () => {
        fillMatrix(topBg);
        fillMatrix(bottomBg);
    });
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', generateMatrixBackground);
} else {
    generateMatrixBackground();
}
