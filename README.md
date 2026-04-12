# Morse Chatbot

A chatbot you can only communicate with by tapping Morse code. The browser converts your taps into text, sends it to the server, and the AI reply is rendered back as Morse code with the decoded text shown beneath.

## How it works

```
[keyboard/button taps]
        ↓
  morse.js decodes dots & dashes → plain text
        ↓
  POST /chat  { messages: [...] }
        ↓
  OpenAI-compatible API (gpt-4o or any model)
        ↓
  reply rendered as Morse + decoded text
```

## Tapping guide

| Action | Timing |
|--------|--------|
| Dot | tap < 200 ms |
| Dash | tap ≥ 200 ms |
| Next letter | pause 700 ms |
| Word space | pause 1.6 s |
| Send message | press **Enter** |

You can tap with the on-screen **TAP** button or the **Space bar**.

## Setup

### Prerequisites

- Node.js 18+
- An API key for OpenAI or any OpenAI-compatible provider (Ollama, Together, Groq, etc.)

### Install

```bash
cd morse-chatbot
npm install
```

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | yes | — | API key |
| `OPENAI_BASE_URL` | no | `https://api.openai.com/v1` | Base URL for any OpenAI-compatible endpoint |
| `OPENAI_MODEL` | no | `gpt-4o` | Model name |
| `PORT` | no | `3000` | Server port |

### Run

```bash
# OpenAI
OPENAI_API_KEY=sk-... npm start

# Local Ollama
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama OPENAI_MODEL=llama3 npm start

# Groq
OPENAI_BASE_URL=https://api.groq.com/openai/v1 OPENAI_API_KEY=gsk_... OPENAI_MODEL=llama3-8b-8192 npm start
```

Then open `http://localhost:3000`.

### Dev (auto-restart on file changes)

```bash
OPENAI_API_KEY=sk-... npm run dev
```

## Project structure

```
morse-chatbot/
├── server.js          # Express app entry point
├── routes/
│   └── chat.js        # POST /chat – calls OpenAI-compatible API
└── public/
    ├── index.html
    ├── style.css
    └── js/
        ├── app.js     # Main UI controller
        ├── morse.js   # Morse alphabet + encoder/decoder
        ├── tap.js     # Tap timing → dot/dash detection
        ├── audio.js   # Web Audio API tone generation
        ├── waveform.js# Canvas waveform visualiser
        └── chat.js    # Chat bubble rendering
```

## API

**`POST /chat`**

Request body:
```json
{
  "messages": [
    { "role": "user",      "content": "HELLO" },
    { "role": "assistant", "content": "ROGER THAT HELLO TO YOU TOO" },
    { "role": "user",      "content": "WHATS THE WEATHER" }
  ]
}
```

Response:
```json
{ "reply": "FAIR SKIES REPORTED OVER MOST REGIONS OVER" }
```
