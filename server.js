// index.ts or app.ts
import express from "express";
import router from "./routes/chat.js";
import { config } from "dotenv";
import langgraphRoutes from "./routes/agent.js";

config();

const app = express();
app.use(express.json());
app.use("/api", router);
app.use("/api/graph", langgraphRoutes);

app.listen(3000, () => console.log("Server on http://localhost:3000"));
