/**
 * Main application entry point.
 * Wires together: tap detector, morse codec, audio, waveform, chat UI, and API calls.
 */

import { morseToText, textToMorse } from './morse.js';
import { beep, playMorse }          from './audio.js';
import { createWaveform }           from './waveform.js';
import { createTapDetector }        from './tap.js';
import { appendMessage, appendThinking } from './chat.js';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const morseDisplay   = document.getElementById('morse-display');
const decodedDisplay = document.getElementById('decoded-display');
const sendBtn        = document.getElementById('send-btn');
const clearBtn       = document.getElementById('clear-btn');
const statusEl       = document.getElementById('status');
const tapKey         = document.getElementById('tap-key');

// ── State ─────────────────────────────────────────────────────────────────────
let currentSymbol = '';  // dots/dashes not yet committed as a letter
let currentMorse  = '';  // full morse string for the current message
const history     = []; // {role, content} pairs sent to the API

// ── Waveform ──────────────────────────────────────────────────────────────────
const waveform = createWaveform(document.getElementById('waveform'));

// ── Display helpers ───────────────────────────────────────────────────────────
function updateDisplay() {
  const preview = currentMorse + (currentSymbol ? ' ' + currentSymbol : '');
  morseDisplay.textContent   = preview || '\u00a0';
  decodedDisplay.textContent = preview.trim()
    ? morseToText(preview.trim())
    : 'decoded text appears here\u2026';
  sendBtn.disabled = !currentMorse.trim() && !currentSymbol;
}

let statusTimer = null;
function setStatus(text, ttlMs = 1000) {
  statusEl.textContent = text;
  clearTimeout(statusTimer);
  if (ttlMs > 0) statusTimer = setTimeout(() => { statusEl.textContent = 'READY'; }, ttlMs);
}

// ── Tap detector ──────────────────────────────────────────────────────────────
const tap = createTapDetector({
  onPressStart() {
    waveform.setActive(true);
    tapKey.classList.add('active');
    beep(30, 700, 0.1);  // soft click feedback
  },
  onPressEnd() {
    waveform.setActive(false);
    tapKey.classList.remove('active');
  },
  onSymbol(sym) {
    currentSymbol += sym;
    setStatus(sym === '.' ? 'DOT' : 'DASH');
    updateDisplay();
    beep(sym === '.' ? 80 : 240, 700);
  },
  onLetterCommit(code) {
    currentMorse  += (currentMorse && !currentMorse.endsWith('/ ') ? ' ' : '') + code;
    currentSymbol  = '';
    updateDisplay();
  },
  onWordSpace() {
    if (currentMorse && !currentMorse.endsWith('/ ')) {
      currentMorse += ' / ';
      updateDisplay();
    }
  },
});

tap.bindKeyboard();
tap.bindButton(tapKey);

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.code === 'Enter') { e.preventDefault(); sendMessage(); }
});

// ── Clear ─────────────────────────────────────────────────────────────────────
clearBtn.addEventListener('click', () => {
  currentSymbol = '';
  currentMorse  = '';
  updateDisplay();
  setStatus('CLEARED');
});

// ── Send ──────────────────────────────────────────────────────────────────────
sendBtn.addEventListener('click', sendMessage);

async function sendMessage() {
  tap.flush();
  const morse = currentMorse.trim();
  if (!morse) return;

  const userText = morseToText(morse);
  appendMessage('user', morse, userText);
  history.push({ role: 'user', content: userText });

  currentSymbol = '';
  currentMorse  = '';
  updateDisplay();
  sendBtn.disabled = true;
  setStatus('SENDING\u2026', 0);

  const thinking = appendThinking();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: history }),
    });

    if (!res.ok) throw new Error(await res.text());

    const { reply } = await res.json();
    history.push({ role: 'assistant', content: reply });

    const botMorse = textToMorse(reply);
    thinking.resolve(botMorse, reply);

    setStatus('PLAYING\u2026', 0);
    await playMorse(botMorse);
    setStatus('READY', 0);
    statusEl.textContent = 'READY';

  } catch (err) {
    thinking.remove();
    appendMessage('bot', '... --- ...', 'Error: ' + err.message);
    setStatus('ERROR');
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
updateDisplay();
