// routes/chat.ts
import express from "express";
import { openai } from "../js/lib/openai.js";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import { TavilySearch } from "@langchain/tavily";
import { MemorySaver } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const langgraphRoutes = express.Router();

langgraphRoutes.post("/test", async (req, res) => {
  console.log(req.body);

  // Define the tools for the agent to use
  const agentTools = [new TavilySearch({ maxResults: 3 })];
  const llm = new ChatOpenAI({
    model: "gpt-4.1-nano",
  });

  // Initialize memory to persist state between graph runs
  const agentCheckpointer = new MemorySaver();
  const agent = createReactAgent({
    llm: llm,
    tools: agentTools,
    checkpointSaver: agentCheckpointer,
  });

  // Now it's time to use!
  const agentFinalState = await agent.invoke(
    { messages: [new HumanMessage("what is the current weather in sf")] },
    { configurable: { thread_id: "42" } }
  );

  console.log(
    agentFinalState.messages[agentFinalState.messages.length - 1].content
  );

  const agentNextState = await agent.invoke(
    { messages: [new HumanMessage("what about ny")] },
    { configurable: { thread_id: "42" } }
  );

  let data =
    agentNextState.messages[agentNextState.messages.length - 1].content;

  console.log(data);

  return res.json(data);
});

export default langgraphRoutes;
