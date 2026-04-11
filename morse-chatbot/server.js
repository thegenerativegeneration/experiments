import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join }  from 'path';
import chatRoute from './routes/chat.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

const app  = express();
const PORT = process.env.PORT ?? 3000;

app.use(express.json());
app.use(express.static(join(__dirname, 'public')));
app.use('/chat', chatRoute);

app.listen(PORT, () => {
  console.log(`Morse chatbot → http://localhost:${PORT}`);
});
