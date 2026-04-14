# The Scriptorium

A small medieval chatbot experiment inspired by the Morse chatbot architecture, but re-skinned as a parchment-and-seal ritual for a Middle High German oracle.

## Core interaction

1. Draft your question in the writing desk.
2. Click **Melt Wax**.
3. Drag the **Signet Ring** onto the wax to send.

## Features

- In-browser WebLLM integration (no backend API).
- Middle High German system prompting with three personas:
  - The Knight
  - The Mystic
  - The Fool
- Typed-response illumination effect and drop-cap styling.
- Palimpsest ghost text persisted in local storage.
- Era-themed controls:
  - Chronicle (history panel)
  - Sacristy (settings panel)
  - Torch (burn/clear chat)
- Dynamic mood and astronomy styling:
  - Melancholy humor meter darkens ink/parchment.
  - Lunar phase shifts illumination tone/layout.

## Model

Configured in `js/llm.js`:

- `thegenerativegeneration/mhg-qwen2.5-3b-webllm`

## Structure

```text
.
├── index.html
├── meta.json
├── README.md
├── style.css
└── js/
    ├── app.js
    ├── chat.js
    └── llm.js
```
