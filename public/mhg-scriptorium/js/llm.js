import * as webllm from 'https://esm.run/@mlc-ai/web-llm';

const MODEL = 'thegenerativegeneration/mhg-qwen2.5-3b-webllm';

const PERSONA_PROMPTS = {
  knight: 'Du bist ein rîterlîcher râtgebære: tugenthafter, kurz, triuwe.',
  mystic: 'Du bist ein mystischer wêgwîser: bildeclich, diuot, zeichenhaft.',
  fool: 'Du bist ein kluoc tôr: spilndiu sprüche mit list und wîsheit.',
};

const BASE_PROMPT =
  'Du bist "der geist in der dinstman des buoches", ein mittelalterlich orakel. ' +
  'Antworte ausschließlîche in Mittelhochdeutsch. ' +
  'Niemer sprich modern englisch oder neuhochdeutsch. ' +
  'Antworte knapp (1-4 sätze), höfisch und klar. Kein Markdown.';

let engine = null;

export async function initLLM(onProgress) {
  engine = await webllm.CreateMLCEngine(MODEL, {
    initProgressCallback: onProgress,
  });
}

function stripThinkTags(text) {
  return text
    .replace(/<think>[\s\S]*?<\/think>/g, '')
    .replace(/<think>[\s\S]*/g, '')
    .trim();
}

export async function chat(messages, persona) {
  if (!engine) throw new Error('LLM not initialised');
  const response = await engine.chat.completions.create({
    messages: [
      { role: 'system', content: `${BASE_PROMPT} ${PERSONA_PROMPTS[persona] ?? PERSONA_PROMPTS.knight}` },
      ...messages,
    ],
    temperature: 0.85,
    max_tokens: 220,
  });

  return stripThinkTags(response.choices[0]?.message?.content ?? '');
}
