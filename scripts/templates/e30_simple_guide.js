(() => {
  'use strict';
  const notebookSource = JSON.parse(document.getElementById('notebook-data').textContent);
  const notebook = JSON.parse(notebookSource);
  const sources = new Map(notebook.cells.map((cell, index) => [String(index + 1), Array.isArray(cell.source) ? cell.source.join('') : cell.source]));
  const toast = document.getElementById('toast');
  toast.hidden = true;
  let toastTimer;
  function announce(message) {
    toast.textContent = message; toast.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { toast.hidden = true; }, 2300);
  }
  function download(name, body, type) {
    const url = URL.createObjectURL(new Blob([body], {type}));
    const link = document.createElement('a'); link.href = url; link.download = name;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  document.getElementById('download-notebook').addEventListener('click', event => {
    event.preventDefault();
    download(notebook.metadata.colab.name, notebookSource, 'application/x-ipynb+json');
    announce('Notebook E30 lengkap diunduh.');
  });
  document.getElementById('download-html').addEventListener('click', () => {
    download('Panduan_E30_MLP.html', '<!doctype html>\n' + document.documentElement.outerHTML, 'text/html;charset=utf-8');
    announce('HTML siap dibaca offline.');
  });
  document.querySelectorAll('[data-copy-cell]').forEach(button => {
    button.addEventListener('click', async () => {
      const source = sources.get(button.dataset.copyCell);
      const text = button.dataset.start ? source.split('\n').slice(Number(button.dataset.start) - 1, Number(button.dataset.end)).join('\n') + '\n' : source;
      try {
        if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
        else {
          const area = document.createElement('textarea'); area.value = text; area.readOnly = true;
          area.style.cssText = 'position:fixed;left:-9999px;top:0'; document.body.append(area); area.select();
          const success = document.execCommand('copy'); area.remove();
          if (!success) throw new Error('copy unavailable');
        }
        announce('Kode asli disalin.');
      } catch { announce('Pilih teks kode lalu salin melalui menu browser.'); }
    });
  });
  document.querySelectorAll('[data-tabs]').forEach(group => {
    const buttons = [...group.querySelectorAll('[role=tab]')];
    function select(button) {
      buttons.forEach(item => {
        const active = item === button; item.setAttribute('aria-selected', String(active)); item.tabIndex = active ? 0 : -1;
        document.getElementById(item.getAttribute('aria-controls')).hidden = !active;
      });
    }
    buttons.forEach((button, index) => {
      button.addEventListener('click', () => select(button));
      button.addEventListener('keydown', event => {
        let next;
        if (event.key === 'ArrowRight') next = (index + 1) % buttons.length;
        if (event.key === 'ArrowLeft') next = (index - 1 + buttons.length) % buttons.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = buttons.length - 1;
        if (next !== undefined) { event.preventDefault(); select(buttons[next]); buttons[next].focus(); }
      });
    });
  });
  const search = document.getElementById('code-search');
  const cards = [...document.querySelectorAll('.code-card')];
  const normalize = value => value.toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '');
  const searchText = new Map(cards.map(card => [card, normalize(card.textContent + ' ' + sources.get(card.dataset.cell))]));
  function filter() {
    const terms = normalize(search.value.trim()).split(/\s+/).filter(Boolean);
    let count = 0;
    cards.forEach(card => { card.hidden = !terms.every(term => searchText.get(card).includes(term)); if (!card.hidden) count++; });
    document.querySelectorAll('.code-chapter').forEach(chapter => { chapter.hidden = !chapter.querySelector('.code-card:not([hidden])'); });
    document.getElementById('code-count').textContent = `${count} dari ${cards.length} cell kode`;
    document.getElementById('empty-search').hidden = count > 0;
    if (terms.length) document.getElementById('code-catalog').open = true;
  }
  search.addEventListener('input', filter);
  search.addEventListener('keydown', event => { if (event.key === 'Escape') { search.value = ''; filter(); } });
  function revealHash() {
    const id = decodeURIComponent(location.hash.slice(1));
    if (!/^cell-\d+$/.test(id)) return;
    const card = document.getElementById(id);
    if (!card) return;
    search.value = ''; filter(); document.getElementById('code-catalog').open = true; card.open = true;
    requestAnimationFrame(() => card.scrollIntoView({block:'start', behavior:'auto'}));
  }
  window.addEventListener('hashchange', revealHash); revealHash(); filter();
  document.querySelectorAll('a[href^="#cell-"]').forEach(link => {
    link.addEventListener('click', () => { if (link.hash === location.hash) revealHash(); });
  });
  if ('IntersectionObserver' in window) {
    const links = [...document.querySelectorAll('nav a')];
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        links.forEach(link => { const active = link.hash === '#' + entry.target.id; link.classList.toggle('active', active); if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current'); });
      });
    }, {rootMargin:'-10% 0px -70% 0px', threshold:0});
    document.querySelectorAll('main>section').forEach(section => observer.observe(section));
  }
  window.E30Guide = Object.freeze({codeCellCount: cards.length, sourceSha256: document.documentElement.dataset.sourceSha256, scenarios:['S1','S2','S3']});
})();
