# Repository Overview

- `morse-chatbot` (served from `public/morse-chatbot`)
- `mhg-finetune` — fine-tuning pipeline for a Middle High German conversational LLM

## Purpose

The root of the repo provides minimal infrastructure to serve static experiment assets:

- `server.js`: Express static server for the `public/` directory
- `package.json`: Node scripts and dependencies for local development
- `public/`: the experiments and all client code

## Structure

```text
.
├── package.json
├── package-lock.json
├── server.js
├── public/          ← browser experiments served by Express
└── mhg-finetune/    ← Python ML pipeline (data, training, evaluation)
```

## Run

```bash
npm install
npm start
```

Open:

- `http://localhost:3000/morse-chatbot/`

## Development

```bash
npm run dev
```

