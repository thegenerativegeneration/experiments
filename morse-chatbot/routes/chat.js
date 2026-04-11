/**
 * POST /chat
 * Accepts a conversation history, forwards it to an OpenAI-compatible API,
 * and returns the reply.
 *
 * Environment variables:
 *   OPENAI_API_KEY  – required
 *   OPENAI_BASE_URL – optional, defaults to https://api.openai.com/v1
 *   OPENAI_MODEL    – optional, defaults to gpt-4o
 */

import OpenAI from 'openai';
import { Router } from 'express';

const router = Router();
const client = new OpenAI({
  apiKey:  process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL, // undefined → library default
});

const MODEL = process.env.OPENAI_MODEL ?? 'gpt-4o';

const SYSTEM_PROMPT = `You are a telegraph operator AI assistant. \
The user communicates with you exclusively via Morse code, and your responses will be \
converted back to Morse code and transmitted to them.

Rules:
- Keep responses SHORT — ideally under 20 words.
- Long words and punctuation are costly to tap, so be concise.
- No markdown, no bullet points, no special characters except . , ? ! -
- You may occasionally use radio/telegraph lingo (COPY THAT, ROGER, OVER) for flavour.
- Be knowledgeable, friendly, and slightly old-fashioned in vocabulary.`;

router.post('/', async (req, res) => {
  const { messages } = req.body;

  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'messages array is required' });
  }

  try {
    const response = await client.chat.completions.create({
      model: MODEL,
      max_tokens: 256,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        ...messages,
      ],
    });

    const reply = response.choices[0].message.content.trim().toUpperCase();
    res.json({ reply });
  } catch (err) {
    console.error('[chat route]', err.message);
    res.status(500).json({ error: err.message });
  }
});

export default router;
