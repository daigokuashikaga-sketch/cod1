/* The orb. Canvas 2D on purpose: no build step, no CDN, works offline, and it
 * renders fine inside a transparent always-on-top Tauri window.
 *
 * The Python core is the source of truth. This file only reflects state that
 * arrives on the /api/events SSE stream; it never decides anything itself.
 */

const PALETTE = {
  idle:      { core: [ 90, 200, 255], ring: [ 60, 140, 220], speed: 0.45, spin: 0.12 },
  listening: { core: [120, 255, 190], ring: [ 60, 200, 150], speed: 1.10, spin: 0.35 },
  thinking:  { core: [255, 200, 110], ring: [230, 150,  60], speed: 1.80, spin: 0.90 },
  speaking:  { core: [140, 230, 255], ring: [ 90, 180, 255], speed: 2.40, spin: 0.55 },
  cancelled: { core: [190, 190, 200], ring: [130, 130, 140], speed: 0.60, spin: 0.10 },
  error:     { core: [255, 120, 120], ring: [210,  70,  70], speed: 2.00, spin: 0.60 },
  paused:    { core: [130, 140, 155], ring: [ 80,  90, 105], speed: 0.18, spin: 0.03 },
};

const canvas = document.getElementById('orb');
const ctx = canvas.getContext('2d');
const stateLabel = document.getElementById('stateLabel');
const dot = document.getElementById('dot');
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');

let state = 'idle';
let energy = 0;          // 0..1, decays; spikes on activity so the orb reacts
const particles = Array.from({ length: 84 }, (_, i) => ({
  angle: (i / 84) * Math.PI * 2,
  radius: 0.62 + Math.random() * 0.3,
  drift: 0.4 + Math.random() * 0.9,
  size: 0.6 + Math.random() * 1.8,
}));

const rgba = (c, a) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;
const lerp = (a, b, t) => a + (b - a) * t;

let shown = PALETTE.idle;

function draw(now) {
  const t = now / 1000;
  const target = PALETTE[state] || PALETTE.idle;
  // Ease between palettes so state changes glide instead of snapping.
  shown = {
    core: shown.core.map((v, i) => lerp(v, target.core[i], 0.06)),
    ring: shown.ring.map((v, i) => lerp(v, target.ring[i], 0.06)),
    speed: lerp(shown.speed, target.speed, 0.06),
    spin: lerp(shown.spin, target.spin, 0.06),
  };

  const w = canvas.width, h = canvas.height;
  const cx = w / 2, cy = h / 2;
  const base = Math.min(w, h) * 0.26;
  const pulse = 1 + 0.06 * Math.sin(t * shown.speed * Math.PI) + energy * 0.14;
  const r = base * pulse;

  ctx.clearRect(0, 0, w, h);
  ctx.globalCompositeOperation = 'lighter';

  // Outer halo.
  const halo = ctx.createRadialGradient(cx, cy, r * 0.5, cx, cy, r * 2.6);
  halo.addColorStop(0, rgba(shown.ring, 0.30 + energy * 0.2));
  halo.addColorStop(0.5, rgba(shown.ring, 0.08));
  halo.addColorStop(1, rgba(shown.ring, 0));
  ctx.fillStyle = halo;
  ctx.fillRect(0, 0, w, h);

  // Orbiting particles.
  for (const p of particles) {
    const a = p.angle + t * shown.spin * p.drift;
    const rr = r * (p.radius + 0.05 * Math.sin(t * shown.speed + p.angle * 3));
    const x = cx + Math.cos(a) * rr * 1.35;
    const y = cy + Math.sin(a) * rr * 1.35;
    ctx.beginPath();
    ctx.arc(x, y, p.size * (1 + energy), 0, Math.PI * 2);
    ctx.fillStyle = rgba(shown.core, 0.35 + energy * 0.4);
    ctx.fill();
  }

  // Rotating rings.
  for (let i = 0; i < 3; i++) {
    const rr = r * (1.18 + i * 0.22);
    const sweep = Math.PI * (0.5 + 0.25 * i);
    const start = t * shown.spin * (i % 2 ? -1.6 : 1.2) + i;
    ctx.beginPath();
    ctx.arc(cx, cy, rr, start, start + sweep);
    ctx.strokeStyle = rgba(shown.ring, 0.5 - i * 0.12);
    ctx.lineWidth = 2.2 - i * 0.5;
    ctx.stroke();
  }

  // Core.
  const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
  core.addColorStop(0, rgba(shown.core, 0.95));
  core.addColorStop(0.55, rgba(shown.core, 0.35));
  core.addColorStop(1, rgba(shown.core, 0));
  ctx.fillStyle = core;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fill();

  ctx.globalCompositeOperation = 'source-over';
  energy = Math.max(0, energy - 0.012);
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

function setState(next) {
  if (!next || next === state) return;
  state = next;
  energy = Math.min(1, energy + 0.5);
  stateLabel.textContent = next;
  const c = PALETTE[next] || PALETTE.idle;
  dot.style.background = rgba(c.core, 1);
  dot.style.boxShadow = `0 0 10px ${rgba(c.core, 0.9)}`;
}

function append(who, body, cls) {
  const row = document.createElement('div');
  row.className = `msg ${cls || who}`;
  row.innerHTML = `<span class="who"></span><span class="body"></span>`;
  row.querySelector('.who').textContent = who;
  row.querySelector('.body').textContent = body;
  log.append(row);
  log.scrollTop = log.scrollHeight;
  while (log.children.length > 200) log.firstChild.remove();
}

const source = new EventSource('/api/events');
source.onmessage = (e) => {
  let ev;
  try { ev = JSON.parse(e.data); } catch { return; }
  switch (ev.kind) {
    case 'state': setState(ev.state); break;
    case 'user': append('you', ev.text); energy = 1; break;
    case 'reply': if (ev.text) append('jarvis', ev.text, 'assistant'); break;
    case 'proactive': if (ev.spoken && ev.text) append('jarvis', ev.text, 'assistant'); break;
    case 'vision':
      if (ev.summary) append('saw', ev.summary, 'vision');
      break;
    case 'tool': append('tool', ev.name + (ev.error ? ' (failed)' : ''), 'system'); break;
    case 'error': append('error', `${ev.where}: ${ev.detail}`, 'system'); break;
    case 'paused': setState(ev.paused ? 'paused' : 'idle'); break;
  }
};
source.onerror = () => stateLabel.textContent = 'disconnected';

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  energy = 1;
  try {
    await fetch('/api/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
  } catch {
    append('error', 'could not reach the Jarvis core', 'system');
  }
});

document.getElementById('pause').addEventListener('click', async (e) => {
  const pausing = state !== 'paused';
  await fetch(pausing ? '/api/pause' : '/api/resume', { method: 'POST' });
  e.target.textContent = pausing ? 'Resume' : 'Pause';
});

document.getElementById('bg').addEventListener('click', () => document.body.classList.toggle('solid'));

async function poll() {
  try {
    const status = await (await fetch('/api/status')).json();
    setState(status.state);
    document.getElementById('spend').textContent = '$' + Number(status.spend_today).toFixed(4);
    document.getElementById('turns').textContent = status.turns;
    document.getElementById('idle').textContent = Math.round(status.idle_for) + 's';
  } catch { /* the core is down; the SSE handler already says so */ }
  setTimeout(poll, 3000);
}
poll();
