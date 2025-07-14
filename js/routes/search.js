import { Router, json } from "express";
import {
  debugVectorStore,
  getAnswer,
  healthCheck,
  searchSimilar,
} from "../js/controllers/customSearch.js";

const searchRouter = Router();

// Middleware for JSON parsing
searchRouter.use(json());

// Route to get AI-generated answer based on vector search
searchRouter.post("/ask", getAnswer);

// Route for similarity search only (no LLM)
searchRouter.post("/search", searchSimilar);

// Health check route
searchRouter.get("/health", healthCheck);
searchRouter.get("/debug", debugVectorStore);

export default searchRouter;

// Example usage in your main app.js:
/*
const express = require('express');
const vectorRoutes = require('./routes/vectorRoutes');

const app = express();
app.use('/api/vectors', vectorRoutes);

app.listen(3000, () => {
  console.log('Server running on port 3000');
});
*/
