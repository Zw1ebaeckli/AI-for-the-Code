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
            frag.appendChild(cell);
        }
        bg.appendChild(frag);
    } else if (current > needed) {
        for (let i = current - 1; i >= needed; i--) {
            bg.removeChild(bg.lastChild);
        }
    }

    // Mask areas behind UI elements: skip numbers where content sits (logo + nav)
    const maskEls = Array.from(document.querySelectorAll('.matrix-mask'));
    const masks = maskEls.map((el) => {
      const r = el.getBoundingClientRect();
      const pad = 12; // give a small buffer so numbers don't hug the edges
      return {
        left: r.left - pad,
        right: r.right + pad,
        top: r.top - pad,
        bottom: r.bottom + pad,
      };
    });

    // Background rect for proper coordinate alignment
    const bgRect = bg.getBoundingClientRect();

    for (let i = 0; i < needed; i++) {
        const cell = bg.children[i];
        const col = i % cols;
        const row = Math.floor(i / cols);
        const cx = bgRect.left + (col + 0.5) * colW;
        const cy = bgRect.top + (row + 0.5) * rowH;

        const isMasked = masks.some((rect) =>
          cx >= rect.left && cx <= rect.right && cy >= rect.top && cy <= rect.bottom
        );

        cell.textContent = isMasked ? '' : String(Math.floor(Math.random() * 10));
    }
  }

      const scheduleFill = () => window.requestAnimationFrame(fillMatrix);

      // Run after layout and on resize
      if (document.readyState === 'complete') {
        scheduleFill();
      } else {
        window.addEventListener('load', scheduleFill, { once: true });
      }

      window.addEventListener('resize', scheduleFill);
})();
