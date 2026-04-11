/**
 * In-browser LLM via WebLLM (WebGPU).
 * Replaces the Express /chat backend — no server needed.
 */

import * as webllm from 'https://esm.run/@mlc-ai/web-llm';

const MODEL = 'SmolLM2-360M-Instruct-q4f16_1-MLC';

const SYSTEM_PROMPT =
  'You are a telegraph operator AI assistant. ' +
  'The user communicates with you exclusively via Morse code, and your responses will be ' +
  'converted back to Morse code and transmitted to them.\n\n' +
  'Rules:\n' +
  '- Keep responses SHORT — ideally under 20 words.\n' +
  '- Long words and punctuation are costly to tap, so be concise.\n' +
  '- No markdown, no bullet points, no special characters except . , ? ! -\n' +
  '- You may occasionally use radio/telegraph lingo (COPY THAT, ROGER, OVER) for flavour.\n' +
  '- Be knowledgeable, friendly, and slightly old-fashioned in vocabulary.';

/** @type {webllm.MLCEngine|null} */
let engine = null;

/**
 * Download and initialise the model.
 * @param {(report: {progress: number, text: string}) => void} onProgress
 */
export async function initLLM(onProgress) {
  engine = await webllm.CreateMLCEngine(MODEL, {
    initProgressCallback: onProgress,
  });
}

/**
 * Run a chat completion against the loaded model.
 * @param {{ role: string, content: string }[]} messages  conversation history
 * @returns {Promise<string>} uppercased reply text
 */
export async function chat(messages) {
  if (!engine) throw new Error('LLM not initialised');

  const response = await engine.chat.completions.create({
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      ...messages,
    ],
    max_tokens: 256,
  });

  return response.choices[0].message.content.trim().toUpperCase();
}
