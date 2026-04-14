import { initLLM, chat } from './llm.js';
import { appendMessage, appendAssistantTyped, appendThinking } from './chat.js';

const statusEl = document.getElementById('status');
const inputEl = document.getElementById('scribe-input');
const waxPotBtn = document.getElementById('wax-pot');
const sealTarget = document.getElementById('seal-target');
const waxBlob = document.getElementById('wax-blob');
const ringEl = document.getElementById('signet-ring');
const torchBtn = document.getElementById('torch-btn');
const chronicleBtn = document.getElementById('chronicle-btn');
const keyBtn = document.getElementById('key-btn');
const chroniclePanel = document.getElementById('chronicle-panel');
const settingsPanel = document.getElementById('settings-panel');
const chronicleList = document.getElementById('chronicle-list');
const ghostLayer = document.getElementById('ghost-layer');
const humorBar = document.getElementById('humor-bar');
const starsNote = document.getElementById('stars-note');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingBar = document.getElementById('loading-bar');
const loadingText = document.getElementById('loading-text');
const inkDrop = document.getElementById('ink-drop');
const desk = document.getElementById('desk');

let waxArmed = false;
let isBusy = false;
let personality = 'knight';
let melancholy = 10;
let idleTimer = null;
const history = [];

const GHOST_KEY = 'mhg-scriptorium-ghost-text';

function setStatus(text) {
  statusEl.textContent = text;
}

function getLunarPhase(date = new Date()) {
  const lunarCycleSeconds = 2551443;
  const nowSec = date.getTime() / 1000;
  const newMoon = new Date('1970-01-07T20:35:00Z').getTime() / 1000;
  const phase = ((nowSec - newMoon) % lunarCycleSeconds) / lunarCycleSeconds;
  const index = Math.floor(phase * 8) % 8;
  return [
    'New Moon',
    'Waxing Crescent',
    'First Quarter',
    'Waxing Gibbous',
    'Full Moon',
    'Waning Gibbous',
    'Last Quarter',
    'Waning Crescent',
  ][index];
}

function applyLunarInfluence() {
  const phase = getLunarPhase();
  const root = document.documentElement;
  const influences = {
    'New Moon': { glow: '#927f5f', shift: '-4px' },
    'Waxing Crescent': { glow: '#a78f5e', shift: '-1px' },
    'First Quarter': { glow: '#b1944f', shift: '3px' },
    'Waxing Gibbous': { glow: '#c5a45b', shift: '2px' },
    'Full Moon': { glow: '#e0c982', shift: '0px' },
    'Waning Gibbous': { glow: '#c2a96f', shift: '-2px' },
    'Last Quarter': { glow: '#ae9263', shift: '4px' },
    'Waning Crescent': { glow: '#9f8759', shift: '1px' },
  };
  const pick = influences[phase];
  root.style.setProperty('--lunar-glow', pick.glow);
  root.style.setProperty('--lunar-shift', pick.shift);
  starsNote.textContent = `Alignment of the spheres: ${phase}`;
}

function renderGhostText() {
  const ghost = localStorage.getItem(GHOST_KEY) || '';
  ghostLayer.textContent = ghost;
}

function archiveGhostText() {
  const items = [...document.querySelectorAll('#chat .msg .msg-text')]
    .map((el) => el.textContent)
    .filter(Boolean)
    .join(' · ');
  if (!items) return;
  const prev = localStorage.getItem(GHOST_KEY) || '';
  const combined = `${prev} ${items}`.trim().slice(-1400);
  localStorage.setItem(GHOST_KEY, combined);
  renderGhostText();
}

function addChronicleEntry(text) {
  const item = document.createElement('div');
  item.className = 'chronicle-item';
  item.textContent = text;
  chronicleList.prepend(item);
}

function applyHumors() {
  const value = Math.max(0, Math.min(100, melancholy));
  humorBar.style.width = `${value}%`;
  const root = document.documentElement;
  if (value > 66) {
    root.style.setProperty('--ink', '#3b3a3f');
    root.style.setProperty('--parchment', '#c2b59a');
    root.style.setProperty('--parchment-dark', '#9e9176');
  } else if (value > 33) {
    root.style.setProperty('--ink', '#31261c');
    root.style.setProperty('--parchment', '#d1bf99');
    root.style.setProperty('--parchment-dark', '#b59b72');
  } else {
    root.style.setProperty('--ink', '#2a1b13');
    root.style.setProperty('--parchment', '#d6c29a');
    root.style.setProperty('--parchment-dark', '#bca072');
  }
}

function updateMelancholy(input) {
  const lowWords = ['trûren', 'leit', 'jâmer', 'tot', 'sünde', 'dunkel', 'pain', 'sorrow'];
  const highWords = ['vreude', 'heil', 'sælde', 'liebe', 'sunne', 'vröude', 'joy'];
  const t = input.toLowerCase();
  if (lowWords.some((w) => t.includes(w))) melancholy += 14;
  else if (highWords.some((w) => t.includes(w))) melancholy -= 12;
  else melancholy += 2;
  applyHumors();
}

function resetSeal() {
  waxArmed = false;
  waxBlob.classList.remove('visible');
  sealTarget.classList.remove('ready');
  setStatus('Draft thy query and seal it.');
}

function armSeal() {
  if (isBusy) return;
  const text = inputEl.value.trim();
  if (!text) {
    setStatus('No ink upon the desk.');
    return;
  }
  waxArmed = true;
  waxBlob.classList.add('visible');
  sealTarget.classList.add('ready');
  setStatus('Wax is laid. Stamp the seal.');
}

async function sendStampedMessage() {
  if (!waxArmed || isBusy) return;
  const userText = inputEl.value.trim();
  if (!userText) return;

  isBusy = true;
  setStatus('The oracle considers...');
  appendMessage('user', userText);
  addChronicleEntry(`You: ${userText.slice(0, 80)}`);
  history.push({ role: 'user', content: userText });
  updateMelancholy(userText);
  inputEl.value = '';
  resetSeal();

  const thinking = appendThinking();
  try {
    const reply = await chat(history, personality);
    thinking.remove();
    history.push({ role: 'assistant', content: reply });
    await appendAssistantTyped(reply);
    addChronicleEntry(`Oracle: ${reply.slice(0, 80)}`);
    updateMelancholy(reply);
    setStatus('Sælde und heil.');
  } catch (error) {
    thinking.remove();
    appendMessage('assistant', `Ein trüebe stœre: ${error.message}`);
    history.pop();
    setStatus('Ink blot in the margins.');
  } finally {
    isBusy = false;
  }
}

waxPotBtn.addEventListener('click', armSeal);

ringEl.addEventListener('dragstart', (event) => {
  event.dataTransfer.setData('text/plain', 'signet');
});

sealTarget.addEventListener('dragover', (event) => {
  if (!waxArmed || isBusy) return;
  event.preventDefault();
  sealTarget.classList.add('over');
});

sealTarget.addEventListener('dragleave', () => {
  sealTarget.classList.remove('over');
});

sealTarget.addEventListener('drop', async (event) => {
  event.preventDefault();
  sealTarget.classList.remove('over');
  const token = event.dataTransfer.getData('text/plain');
  if (token !== 'signet') return;
  await sendStampedMessage();
});

ringEl.addEventListener('click', sendStampedMessage);

torchBtn.addEventListener('click', () => {
  archiveGhostText();
  history.length = 0;
  document.body.classList.add('burning');
  document.getElementById('chat').innerHTML = '';
  setTimeout(() => document.body.classList.remove('burning'), 550);
  setStatus('The parchment is burned clean.');
});

chronicleBtn.addEventListener('click', () => {
  chroniclePanel.classList.toggle('collapsed');
});

keyBtn.addEventListener('click', () => {
  settingsPanel.classList.toggle('collapsed');
});

document.querySelectorAll('.persona').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.persona').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    personality = btn.dataset.persona;
    setStatus(`Triptych: ${btn.textContent}`);
  });
});

function resetIdleInkDrop() {
  clearTimeout(idleTimer);
  inkDrop.classList.remove('idle');
  idleTimer = setTimeout(() => inkDrop.classList.add('idle'), 2400);
}

inputEl.addEventListener('input', resetIdleInkDrop);
inputEl.addEventListener('focus', resetIdleInkDrop);
desk.addEventListener('mousemove', resetIdleInkDrop);

applyLunarInfluence();
renderGhostText();
applyHumors();
setStatus('Preparing the oracle...');
resetIdleInkDrop();

inputEl.disabled = true;
waxPotBtn.disabled = true;
ringEl.setAttribute('draggable', 'false');

initLLM((report) => {
  loadingBar.style.width = `${(report.progress * 100).toFixed(1)}%`;
  loadingText.textContent = report.text;
}).then(() => {
  loadingOverlay.classList.add('hidden');
  inputEl.disabled = false;
  waxPotBtn.disabled = false;
  ringEl.setAttribute('draggable', 'true');
  setStatus('Hie bevîndet iuch der geist in der dinstman des buoches.');
  appendMessage('assistant', 'Sælde unde heil, wanderære. Waz ist dîn ger?');
}).catch((error) => {
  loadingText.textContent = `Error: ${error.message}`;
  setStatus('The oracle remains veiled.');
});
