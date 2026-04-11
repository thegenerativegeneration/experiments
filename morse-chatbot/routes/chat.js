/**
 * POST /chat
 * Accepts a conversation history, forwards it to Claude, returns the reply.
 */

import Anthropic from '@anthropic-ai/sdk';
import { Router } from 'express';

const router = Router();
const client = new Anthropic(); // reads ANTHROPIC_API_KEY from env

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
    const response = await client.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 256,
      system: SYSTEM_PROMPT,
      messages,
    });

    const reply = response.content[0].text.trim().toUpperCase();
    res.json({ reply });
  } catch (err) {
    console.error('[chat route]', err.message);
    res.status(500).json({ error: err.message });
  }
});

export default router;
