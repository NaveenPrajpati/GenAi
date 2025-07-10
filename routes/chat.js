// routes/chat.ts
import express from "express";
import { openai } from "../lib/openai.js";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { MongoClient } from "mongodb";
import { config } from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
config();
const loader = new PDFLoader("files/naveenResume.pdf");

const docs = await loader.load();
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

  const llm2 = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0,
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
  const response = await llm2.invoke(promptValue);

  console.log(response.content);

  return res.json(response.content);
});
router.post("/test-template", async (req, res) => {
  console.log(req.body);
  const llm = new ChatOpenAI({
    model: "gpt-4.1-nano",
  });

  const text = " tell me a joke about {topic}";
  //string prompt template
  const promtTemplate = PromptTemplate.fromTemplate(text);
  const temp1 = await promtTemplate.invoke({ topic: "mobile" });

  console.log(temp1.toChatMessages());

  //chat prompt template
  const chatTemplate = ChatPromptTemplate.fromMessages([
    ["system", "you are a helpful assistant"],
    ["user", "tell me a joke about {topic}"],
  ]);

  const temp2 = await chatTemplate.invoke({ topic: "cats" });

  console.log(temp2.toChatMessages());

  const data1 = await llm.invoke(temp1);
  const data2 = await llm.invoke(temp2);

  return res.json({ temp1, temp2, data1, data2 });
});
router.post("/test-search", async (req, res) => {
  console.log(req.body);

  // const textSplitter = new RecursiveCharacterTextSplitter({
  //   chunkSize: 1000,
  //   chunkOverlap: 200,
  // });

  // const allSplits = await textSplitter.splitDocuments(docs);

  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-large",
  });

  // const vector1 = await embeddings.embedQuery(allSplits[0].pageContent);
  // const vector2 = await embeddings.embedQuery(allSplits[1].pageContent);

  // console.assert(vector1.length === vector2.length);
  // console.log(`Generated vectors of length ${vector1.length}\n`);
  // console.log(vector1.slice(0, 10));

  const client = new MongoClient(process.env.MONGO_URI || "");
  client.connect();
  const collection = client.db("genai").collection("vectors");

  console.log("Docs in collection:", await collection.countDocuments());

  const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
    collection: collection,
    indexName: "vector_index",
    textKey: "text",
    embeddingKey: "embedding",
  });

  const data = await vectorStore.similaritySearch(" what is naveen email", 3);

  console.log("Top match:\n", data);
  // await vectorStore.addDocuments(allSplits);

  await client.close();
  return res.json(data);
});

export default router;
