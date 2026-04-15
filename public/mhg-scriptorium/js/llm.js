import * as webllm from 'https://esm.run/@mlc-ai/web-llm';

const MODEL = 'thegenerativegeneration/mhg-qwen2.5-3b-webllm';
const BASE_MODEL_FOR_LIB = 'Qwen2.5-3B-Instruct-q4f16_1-MLC';
const MODEL_BASE_URL = `https://huggingface.co/${MODEL}/resolve/main/`;

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

function createAppConfigWithCustomModel() {
  const prebuilt = webllm.prebuiltAppConfig;
  const baseRecord = prebuilt.model_list.find((entry) => entry.model_id === BASE_MODEL_FOR_LIB);

  if (!baseRecord) {
    throw new Error(`Could not find base WebLLM record for ${BASE_MODEL_FOR_LIB}`);
  }

  const withoutExistingCustom = prebuilt.model_list.filter((entry) => entry.model_id !== MODEL);
  const customRecord = {
    ...baseRecord,
    model_id: MODEL,
    model: MODEL_BASE_URL,
  };

  return {
    ...prebuilt,
    model_list: [...withoutExistingCustom, customRecord],
  };
}

export async function initLLM(onProgress) {
  engine = await webllm.CreateMLCEngine(MODEL, {
    appConfig: createAppConfigWithCustomModel(),
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
