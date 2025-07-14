import express from "express";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { MongoClient } from "mongodb";
import { config } from "dotenv";
config();

// Initialize OpenAI components
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large",
});

const llm = new ChatOpenAI({
  model: "gpt-4.1-nano",
  temperature: 0.7,
});

// MongoDB connection
const client = new MongoClient(process.env.MONGO_URI || "");
const collection = client.db("genai").collection("vectors");

const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
  collection: collection,
  indexName: "vector_index",
  textKey: "text",
  embeddingKey: "embedding",
});

// Prompt template for RAG
const ragPrompt = new PromptTemplate({
  template: `
Use the following context to answer the question. If you cannot find the answer in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Answer:`,
  inputVariables: ["context", "question"],
});

// Controller function
export const getAnswer = async (req, res) => {
  try {
    const { question } = req.body;

    console.log("🔍 Question received:", question);

    if (!question) {
      return res.status(400).json({
        error: "Question is required",
        success: false,
      });
    }

    // Search for similar documents
    console.log("🔍 Searching for similar documents...");
    const similarDocs = await vectorStore.similaritySearch(question, 3);

    console.log("📄 Found documents:", similarDocs.length);
    console.log(
      "📄 First doc preview:",
      similarDocs[0]?.pageContent?.substring(0, 100)
    );

    if (similarDocs.length === 0) {
      return res.status(404).json({
        error: "No relevant documents found",
        success: false,
        debug: {
          question: question,
          searchResults: 0,
        },
      });
    }

    // Combine context from similar documents
    const context = similarDocs.map((doc) => doc.pageContent).join("\n\n");

    console.log("📝 Context length:", context.length);
    console.log("📝 Context preview:", context.substring(0, 200));

    // Generate prompt
    const prompt = await ragPrompt.format({
      context: context,
      question: question,
    });

    console.log("🤖 Generated prompt length:", prompt.length);
    console.log("🤖 Prompt preview:", prompt.substring(0, 300));

    // Get answer from LLM
    console.log("🤖 Calling LLM...");
    const response = await llm.invoke(prompt);

    console.log("✅ LLM Response:", response.content);

    // Return response
    res.json({
      success: true,
      answer: response.content,
      sources: similarDocs.map((doc) => ({
        content: doc.pageContent.substring(0, 200) + "...",
        metadata: doc.metadata,
      })),
      contextUsed: context.length,
      debug: {
        documentsFound: similarDocs.length,
        contextLength: context.length,
        promptLength: prompt.length,
      },
    });
  } catch (error) {
    console.error("❌ Error in getAnswer:", error);
    res.status(500).json({
      error: "Internal server error",
      success: false,
      message: error.message,
      stack: process.env.NODE_ENV === "development" ? error.stack : undefined,
    });
  }
};

// Alternative controller for just similarity search (without LLM)
export const searchSimilar = async (req, res) => {
  try {
    const { query, limit = 3 } = req.body;

    if (!query) {
      return res.status(400).json({
        error: "Query is required",
        success: false,
      });
    }

    const similarDocs = await vectorStore.similaritySearchWithScore(
      query,
      limit
    );

    res.json({
      success: true,
      results: similarDocs.map(([doc, score]) => ({
        content: doc.pageContent,
        metadata: doc.metadata,
        similarity_score: score,
      })),
    });
  } catch (error) {
    console.error("Error in searchSimilar:", error);
    res.status(500).json({
      error: "Internal server error",
      success: false,
      message: error.message,
    });
  }
};

// Health check for vector store
export const healthCheck = async (req, res) => {
  try {
    await client.db("admin").command({ ping: 1 });
    res.json({
      success: true,
      message: "Vector store is healthy",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: "Vector store connection failed",
      error: error.message,
    });
  }
};

export const debugVectorStore = async (req, res) => {
  try {
    // Check total documents in collection
    const totalDocs = await collection.countDocuments();
    console.log("📊 Total documents in collection:", totalDocs);

    // Get a sample document
    const sampleDoc = await collection.findOne();
    console.log("📄 Sample document structure:", {
      keys: Object.keys(sampleDoc || {}),
      hasText: !!sampleDoc?.text,
      hasEmbedding: !!sampleDoc?.embedding,
      embeddingLength: sampleDoc?.embedding?.length,
    });

    // Check if indexes exist
    const indexes = await collection.listIndexes().toArray();
    console.log(
      "📇 Available indexes:",
      indexes.map((idx) => idx.name)
    );

    // Test manual vector search with MongoDB aggregation
    const testEmbedding = await embeddings.embedQuery("test");
    console.log("🔍 Test embedding length:", testEmbedding.length);

    // Manual aggregation pipeline for vector search
    const pipeline = [
      {
        $vectorSearch: {
          index: "vector_index", // Try your actual index name
          path: "embedding",
          queryVector: testEmbedding,
          numCandidates: 100,
          limit: 3,
        },
      },
      {
        $project: {
          text: 1,
          score: { $meta: "vectorSearchScore" },
        },
      },
    ];

    let manualResults = [];
    try {
      manualResults = await collection.aggregate(pipeline).toArray();
      console.log("🔍 Manual vector search results:", manualResults.length);
    } catch (vectorSearchError) {
      console.log("❌ Vector search error:", vectorSearchError.message);
    }

    // Test with alternative index names
    const commonIndexNames = ["vector_index", "default", "vector_search_index"];
    const testResults = {};

    for (const indexName of commonIndexNames) {
      try {
        const testVectorStore = new MongoDBAtlasVectorSearch(embeddings, {
          collection: collection,
          indexName: indexName,
          textKey: "text",
          embeddingKey: "embedding",
        });

        const results = await testVectorStore.similaritySearch("test", 1);
        testResults[indexName] = results.length;
        console.log(`🔍 Index "${indexName}" results:`, results.length);
      } catch (error) {
        testResults[indexName] = `Error: ${error.message}`;
        console.log(`❌ Index "${indexName}" error:`, error.message);
      }
    }

    res.json({
      success: true,
      vectorStore: {
        totalDocuments: totalDocs,
        sampleDocument: {
          keys: Object.keys(sampleDoc || {}),
          hasText: !!sampleDoc?.text,
          hasEmbedding: !!sampleDoc?.embedding,
          embeddingLength: sampleDoc?.embedding?.length,
          textPreview: sampleDoc?.text?.substring(0, 100),
        },
        indexes: indexes.map((idx) => ({
          name: idx.name,
          key: idx.key,
        })),
        manualVectorSearch: {
          resultsFound: manualResults.length,
          firstResult: manualResults[0]?.text?.substring(0, 100),
        },
        indexTests: testResults,
        testEmbeddingLength: testEmbedding.length,
      },
    });
  } catch (error) {
    console.error("❌ Error in debugVectorStore:", error);
    res.status(500).json({
      success: false,
      error: error.message,
      stack: process.env.NODE_ENV === "development" ? error.stack : undefined,
    });
  }
};
