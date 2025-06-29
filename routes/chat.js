// routes/chat.ts
import express from "express";
import { openai } from "../lib/openai.js";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";

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
router.post("/test", async (req, res) => {
  console.log(req.body);

  const llm = new ChatOpenAI({
    model: "gpt-4.1-nano",
  });

  //this language model approach to call llm

  // const messages = [
  //   new SystemMessage("Translate the following from English into hindi"),
  //   new HumanMessage("hi!"),
  // ];
  // await llm.invoke(messages);

  const systemTemplate = "Translate the following from English into {language}";

  const promptTemplate = ChatPromptTemplate.fromMessages([
    ["system", systemTemplate],
    ["user", "{text}"],
  ]);

  const promptValue = await promptTemplate.invoke({
    language: "hindi",
    text: "hello",
  });
  console.log(promptValue.toChatMessages());
  const response = await llm.invoke(promptValue);

  console.log(response.content);

  return res.json(response.content);
});

export default router;
