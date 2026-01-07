// Procedurally generate numeric matrix background
(function() {
  const root = document.documentElement;
  const bg = document.querySelector('.matrix-background');
  if (!bg) return;

  function getVar(name, fallback) {
    const v = getComputedStyle(root).getPropertyValue(name).trim();
    if (!v) return fallback;
    if (v.endsWith('px')) return parseFloat(v);
    const num = parseFloat(v);
    return isNaN(num) ? fallback : num;
  }

  function fillMatrix() {
    const colW = getVar('--matrix-col-width', 64);
    const rowH = getVar('--matrix-line-height', 28);

    const w = window.innerWidth;
    const h = window.innerHeight;
    const cols = Math.max(1, Math.floor(w / colW));
    const rows = Math.max(1, Math.floor(h / rowH));

    // Configure grid
    bg.style.gridTemplateColumns = `repeat(${cols}, ${colW}px)`;
    bg.style.gridAutoRows = `${rowH}px`;

    const needed = cols * rows;
    const current = bg.childElementCount;

    // Add/remove to match needed cells
    if (current < needed) {
      const frag = document.createDocumentFragment();
      for (let i = current; i < needed; i++) {
        const cell = document.createElement('div');
        cell.className = 'matrix-cell';
        cell.textContent = String(Math.floor(Math.random() * 10));
        frag.appendChild(cell);
      }
      bg.appendChild(frag);
    } else if (current > needed) {
      for (let i = current - 1; i >= needed; i--) {
        bg.removeChild(bg.lastChild);
      }
    }
  }

  fillMatrix();
  window.addEventListener('resize', fillMatrix);
})();
