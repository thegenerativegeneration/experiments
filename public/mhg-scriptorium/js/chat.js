const chatEl = document.getElementById('chat');

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function createMessage(role, text) {
  const wrap = document.createElement('article');
  wrap.className = `msg ${role}`;

  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = role === 'user' ? 'Bitter' : 'Orakel';

  const textEl = document.createElement('div');
  textEl.className = 'msg-text';
  textEl.textContent = text;

  wrap.append(label, textEl);
  chatEl.appendChild(wrap);
  scrollToBottom();
  return { wrap, textEl };
}

export function appendMessage(role, text) {
  createMessage(role, text);
}

export async function appendAssistantTyped(text, delayMs = 17) {
  const { textEl } = createMessage('assistant', '');
  for (let i = 0; i < text.length; i += 1) {
    textEl.textContent += text[i];
    if (i % 2 === 0) scrollToBottom();
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }
  scrollToBottom();
}

export function appendThinking() {
  const { wrap } = createMessage('assistant', 'Diu feder kratzt in swîgen...');
  wrap.classList.add('thinking');
  return {
    remove() { wrap.remove(); },
  };
}
