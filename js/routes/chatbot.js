/**
 * Complete LangChain ➜ LangGraph ➜ Express controller
 * ----------------------------------------------------
 * This single file contains **everything** from the official LangChain
 * chatbot tutorial, wired into an Express route so you can POST user
 * messages and get JSON replies while *persisting* conversation threads.
 *
 * ✔️  Model & prompt setup
 * ✔️  LangGraph state + checkpoint (MemorySaver by default)
 * ✔️  Message‑window trimming
 * ✔️  Thread‑scoped persistence via `threadId`
 *
 * How to run
 * ----------
 * 1.   npm i express @types/express @langchain/openai @langchain/core @langchain/langgraph uuid dotenv
 * 2.   export OPENAI_API_KEY="sk‑..."        # or use a .env file
 * 3.   ts-node chatController.ts              # or compile & node build
 * 4.   curl -X POST http://localhost:3000/chat \
 *        -H "Content-Type: application/json"  \
 *        -d '{"message":"Hi, I am Bob"}'
 *
 * Swap `MemorySaver` for RedisSaver / PostgresSaver / S3Saver if you need
 * cross‑process or cross‑restart durability (see comments below).
 */

import "dotenv/config";
import express, { Router, Request, Response } from "express";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { HumanMessage, BaseMessage } from "@langchain/core/messages";
import { StateGraph, START, END } from "@langchain/langgraph";
import { MessagesAnnotation, GraphAnnotation } from "@langchain/langgraph/prebuilt";
import { MemorySaver /* RedisSaver, PostgresSaver, S3Saver */ } from "@langchain/langgraph/checkpoint";
import { trimMessages } from "@langchain/core/utils";
import { v4 as uuidv4 } from "uuid";

/**
 * ------------------------------------------------------------------
 * 1⃣  Build the chatbot exactly like the tutorial
 * ------------------------------------------------------------------
 */

/* 1. Chat model */
const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0.7 });

/* 2. Prompt template */
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant that responds in {language}."],
  new MessagesPlaceholder("messages"),
]);

/* 3. History trimmer */
const trimmer = trimMessages({ maxTokens: 1500, strategy: "last", includeSystem: true });

/* 4. State definition */
interface BotState extends MessagesAnnotation { language: string }
const annotation = GraphAnnotation.fromRecord<BotState>({ messages: "array", language: "string" });

/* 5. Graph node */
async function chatNode(state: BotState): Promise<Partial<BotState>> {
  const history = trimmer(state.messages);
  const chain = prompt.pipe(llm);
  const assistant: BaseMessage = await chain.invoke({ messages: history, language: state.language });
  return { messages: [...history, assistant] };
}

/* 6. Compile graph with checkpoint */
const workflow = new StateGraph(annotation)
  .addNode("chat", chatNode)
  .addEdge(START, "chat")
  .addEdge("chat", END);

// MemorySaver keeps state in RAM per process. Swap for RedisSaver or others for true persistence.
const checkpointer = new MemorySaver();
const appGraph = workflow.compile({ checkpointer });

/**
 * ------------------------------------------------------------------
 * 2⃣  Express controller & router
 * ------------------------------------------------------------------
 */

interface ChatRequestBody { message: string; threadId?: string; language?: string }

async function chatController(req: Request, res: Response) {
  const { message, threadId, language = "English" } = req.body as ChatRequestBody;
  if (!message || typeof message !== "string") {
    return res.status(400).json({ error: "'message' (string) is required" });
  }
  const tid = threadId || uuidv4();
  try {
    const result = await appGraph.invoke(
      { messages: [new HumanMessage(message)], language },
      { configurable: { thread_id: tid } },
    );
    const reply = result.messages.at(-1)?.content ?? "";
    return res.json({ threadId: tid, reply });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "Chatbot error" });
  }
}

/* 3. Plug‑and‑play router */
const chatRouter = Router();
chatRouter.post("/chat", chatController);

export default chatRouter; // so you can `app.use(chatRouter)` elsewhere


