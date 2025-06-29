// routes/chat.ts
import express from "express";
import { openai } from "../lib/openai.js";

const router = express.Router();

router.post("/chat", async (req, res) => {
  const messages = req.body.messages;

  const start = Date.now();
  let totalTokens = 0;

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.flushHeaders();

  const stream = await openai.chat.completions.create({
    model: "gpt-4.1-nano",
    messages,
    stream: true,
  });

  for await (const chunk of stream) {
    const content = chunk.choices?.[0]?.delta?.content || "";
    totalTokens += content.length / 4; // rough estimate
    res.write(`data: ${content}\n\n`);
  }

  const duration = Date.now() - start;
  res.write(`\ndata: [end]\n`);
  res.end();

  console.log(
    `⏱️ Latency: ${duration}ms, 🧠 Tokens (est.): ${Math.round(totalTokens)}`
  );
});

export default router;
