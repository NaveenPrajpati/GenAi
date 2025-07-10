import { Router } from "express";

const ragRouter = Router();

ragRouter.post("/test", (req, res) => {
  const data = req.body;
});

export default ragRouter;
