"""
INCOMING TICKET → CLASSIFY → FIND SIMILAR → TAKE ACTION

Example Flow:
1. Customer submits: "My payment failed and I can't access premium features"
2. Agent classifies: "Billing" (85%+ confidence)
3. Agent searches: Finds 3 similar past billing tickets
4. Agent takes action: Routes to billing team + sends auto-response

Features:
- Embedding-backed classifier (LogisticRegression)
- FAISS similarity retrieval over past tickets
- Action node: create_github_issue (stub) with idempotency
- LangGraph pipeline + typed state
- CLI: train / index / triage / eval / serve (FastAPI)

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split


# langchain + embeddings
def get_embedder():
    """
    Returns a tuple (name, embedder) with embed_documents(texts) and embed_query(text).
    Selects OpenAI if OPENAI_API_KEY exists or EMBEDDINGS_BACKEND=openai.
    Otherwise uses sentence-transformers (all-MiniLM-L6-v2 by default).
    """
    backend_env = os.getenv("EMBEDDINGS_BACKEND", "auto").lower()
    want_openai = backend_env == "openai" or (
        backend_env == "auto" and os.getenv("OPENAI_API_KEY")
    )
    if want_openai:
        try:
            from langchain_openai import OpenAIEmbeddings

            model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
            return (
                "openai",
                OpenAIEmbeddings(model=model),
            )
        except Exception as e:
            print(
                f"[warn] OpenAI embeddings unavailable ({e}); falling back to sentence-transformers",
                file=sys.stderr,
            )
    # fallback to sentence-transformers
    from sentence_transformers import SentenceTransformer

    st_model = os.getenv("ST_MODEL", "all-MiniLM-L6-v2")
    _model = SentenceTransformer(st_model)

    class _STEmbedder:
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return _model.encode(texts, normalize_embeddings=False).tolist()

        def embed_query(self, text: str) -> List[float]:
            return _model.encode([text], normalize_embeddings=False)[0].tolist()

    return ("sentence-transformers", _STEmbedder())


# --- Types & State ---
TicketLabel = Literal["Bug", "Billing", "How-to"]


class SimilarTicket(BaseModel):
    id: str
    text: str
    label: TicketLabel
    url: Optional[str] = None
    score: float


class ActionResult(BaseModel):
    action: Literal["none", "create_github_issue"]
    success: bool
    issue_id: Optional[str] = None
    existed: bool = False
    details: Dict[str, Any] = {}


class TriageState(TypedDict, total=False):
    ticket_id: str
    ticket_text: str
    predicted_label: TicketLabel
    label_probs: Dict[str, float]
    similar: List[SimilarTicket]
    action_result: ActionResult
    logs: List[str]


# --- Paths ---
DATA_DIR = os.path.join("data")
ARTIFACTS_DIR = os.path.join("artifacts")
CLF_PATH = os.path.join(ARTIFACTS_DIR, "clf.pkl")
FAISS_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")
FAISS_META_PATH = os.path.join(ARTIFACTS_DIR, "faiss_meta.json")
ISSUES_STORE_PATH = os.path.join(ARTIFACTS_DIR, "issues_store.jsonl")
LABELED_CSV = os.path.join(DATA_DIR, "tickets_labeled.csv")
HISTORY_CSV = os.path.join(DATA_DIR, "tickets_history.csv")


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    if not os.path.exists(ISSUES_STORE_PATH):
        open(ISSUES_STORE_PATH, "a").close()


# --- Utilities ---
def _normalize_text(t: str) -> str:
    return " ".join((t or "").strip().split())


def _fingerprint(text: str) -> str:
    norm = _normalize_text(text).lower()
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


def _load_issue_store() -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.exists(ISSUES_STORE_PATH):
        return out
    with open(ISSUES_STORE_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["fingerprint"]] = rec["issue_id"]
            except Exception:
                pass
    return out


def create_github_issue_stub(title: str, body: str) -> Dict[str, Any]:
    """
    Idempotent issue creation stub.
    - Compute fingerprint over body (fallback to title)
    - If exists, return existing {issue_id, existed=True}
    - Else, create new, persist to jsonl, return existed=False
    """
    ensure_dirs()
    fp = _fingerprint(body or title)
    store = _load_issue_store()
    if fp in store:
        return {"issue_id": store[fp], "existed": True}
    issue_id = f"ISSUE-{uuid.uuid4().hex[:8]}"
    rec = {"fingerprint": fp, "issue_id": issue_id, "title": title, "body": body}
    with open(ISSUES_STORE_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return {"issue_id": issue_id, "existed": False}


# --- Classifier train/load/predict ---
def train_classifier(
    csv_path: str,
    clf_out: str,
    eval_split: float = 0.0,
    clf_C: float = 1.0,
) -> None:
    ensure_dirs()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Missing {csv_path}. Create CSV with columns: id,text,label"
        )
    df = pd.read_csv(csv_path)
    assert {"id", "text", "label"}.issubset(
        df.columns
    ), "tickets_labeled.csv must have id,text,label"
    texts = df["text"].astype(str).apply(_normalize_text).tolist()
    y = df["label"].astype(str).tolist()

    emb_name, emb = get_embedder()
    print(f"[embeddings] backend={emb_name}")

    X_all = np.array(emb.embed_documents(texts))

    # Optional holdout eval (not saved)
    if eval_split and 0.0 < eval_split < 0.99:
        Xtr, Xte, ytr, yte = train_test_split(
            X_all, y, test_size=eval_split, stratify=y, random_state=42
        )
        clf_eval = LogisticRegression(max_iter=500, C=clf_C).fit(Xtr, ytr)
        pred = clf_eval.predict(Xte)
        print("\n[Eval split report]")
        print(classification_report(yte, pred, digits=3))
        print("Macro F1:", f1_score(yte, pred, average="macro"))

    # Train on all data for the production model
    clf = LogisticRegression(max_iter=500, C=clf_C).fit(X_all, y)
    import pickle

    with open(clf_out, "wb") as f:
        pickle.dump({"clf": clf, "labels": sorted(set(y)), "emb_name": emb_name}, f)
    print(f"[saved] classifier → {clf_out}")


def _load_clf(clf_path: str):
    import pickle

    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"Missing classifier at {clf_path}. Run 'train' first.")
    obj = pickle.load(open(clf_path, "rb"))
    return obj["clf"]


def classify_text(text: str) -> Dict[str, Any]:
    emb_name, emb = get_embedder()
    vec = np.array(emb.embed_query(text)).reshape(1, -1)
    clf = _load_clf(CLF_PATH)
    proba = clf.predict_proba(vec)[0]
    labels = clf.classes_
    label = labels[np.argmax(proba)]
    probs = {labels[i]: float(proba[i]) for i in range(len(labels))}
    return {"label": str(label), "probs": probs}


# --- FAISS index build/search ---
def build_faiss(csv_path: str) -> None:
    ensure_dirs()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Missing {csv_path}. Create CSV with columns: id,text,label,url,resolution"
        )
    df = pd.read_csv(csv_path)
    assert {"id", "text"}.issubset(
        df.columns
    ), "tickets_history.csv must have at least id,text"
    texts = df["text"].astype(str).apply(_normalize_text).tolist()

    emb_name, emb = get_embedder()
    print(f"[embeddings] backend={emb_name}")
    X = np.array(emb.embed_documents(texts)).astype("float32")

    # lazy import faiss
    import faiss

    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    faiss.write_index(index, FAISS_INDEX_PATH)
    df.to_json(FAISS_META_PATH, orient="records")
    print(f"[saved] FAISS index → {FAISS_INDEX_PATH}")
    print(f"[saved] metadata    → {FAISS_META_PATH}")


def faiss_search(query: str, k: int = 5) -> List[SimilarTicket]:
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_META_PATH):
        raise FileNotFoundError("FAISS artifacts missing. Run 'index' first.")
    # lazy imports
    import faiss

    emb_name, emb = get_embedder()
    q = np.array(emb.embed_query(query)).astype("float32")[None, :]
    faiss.normalize_L2(q)
    index = faiss.read_index(FAISS_INDEX_PATH)
    D, I = index.search(q, k)

    meta = json.load(open(FAISS_META_PATH))
    sims: List[SimilarTicket] = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        m = meta[idx]
        sims.append(
            SimilarTicket(
                id=str(m.get("id")),
                text=str(m.get("text")),
                label=str(m.get("label", "How-to")),
                url=m.get("url"),
                score=float(score),
            )
        )
    return sims


# --- LangGraph pipeline ---
def build_graph():
    from langgraph.graph import StateGraph, START, END

    graph = StateGraph(TriageState)

    @graph.add_node
    def classify(state: TriageState):
        out = classify_text(state["ticket_text"])
        logs = state.get("logs", []) + [f"classify: {out['label']} {out['probs']}"]
        return {
            "predicted_label": out["label"],
            "label_probs": out["probs"],
            "logs": logs,
        }

    @graph.add_node
    def retrieve_similar(state: TriageState):
        sims = faiss_search(state["ticket_text"], k=5)
        logs = state.get("logs", []) + [f"retrieve_similar: k={len(sims)}"]
        return {"similar": [s.model_dump() for s in sims], "logs": logs}

    @graph.add_node
    def create_issue(state: TriageState):
        title = f"[Bug] {state['ticket_text'][:80]}"
        body = state["ticket_text"]
        res = create_github_issue_stub(title, body)
        ar = ActionResult(
            action="create_github_issue",
            success=True,
            issue_id=res["issue_id"],
            existed=res["existed"],
            details={},
        )
        logs = state.get("logs", []) + [
            f"create_issue: {ar.issue_id} existed={ar.existed}"
        ]
        return {"action_result": ar.model_dump(), "logs": logs}

    def should_create_issue(state: TriageState) -> Literal["create_issue", "__end__"]:
        return "create_issue" if state.get("predicted_label") == "Bug" else "__end__"

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve_similar")
    graph.add_conditional_edges(
        "retrieve_similar",
        should_create_issue,
        {"create_issue": "create_issue", "__end__": END},
    )
    graph.add_edge("create_issue", END)

    return graph.compile()


def triage_once(ticket_id: str, text: str) -> TriageState:
    app = build_graph()
    init = {"ticket_id": ticket_id, "ticket_text": _normalize_text(text), "logs": []}
    final: TriageState = app.invoke(init)
    return final


# --- Evaluation harness ---
def evaluate() -> None:
    if not os.path.exists(LABELED_CSV):
        raise FileNotFoundError(f"Missing {LABELED_CSV}")
    df = pd.read_csv(LABELED_CSV)
    y_true, y_pred = [], []
    bug_true_total = 0
    bug_action_ok = 0
    nonbug_action_count = 0

    for r in df.to_dict(orient="records"):
        out = triage_once(str(r["id"]), str(r["text"]))
        y_true.append(str(r["label"]))
        y_pred.append(str(out.get("predicted_label")))
        # Action checks
        if r["label"] == "Bug":
            bug_true_total += 1
            ar = out.get("action_result", {})
            if (
                out.get("predicted_label") == "Bug"
                and ar
                and ar.get("action") == "create_github_issue"
                and ar.get("success")
            ):
                bug_action_ok += 1
        else:
            # ensure we didn't create issues for non-bugs
            ar = out.get("action_result")
            if ar and ar.get("action") == "create_github_issue":
                nonbug_action_count += 1

    print("\n[Classification report]")
    print(classification_report(y_true, y_pred, digits=3))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))

    if bug_true_total:
        print(
            "Bug action correctness:",
            f"{bug_action_ok}/{bug_true_total} (predicted Bug + action ran)",
        )
    if nonbug_action_count:
        print(f"[warn] Non-bug issues created: {nonbug_action_count} (should be 0)")

    # Idempotency check on first bug (if exists)
    bug_rows = [r for r in df.to_dict(orient="records") if str(r["label"]) == "Bug"]
    if bug_rows:
        sample = bug_rows[0]
        r1 = triage_once(str(sample["id"]), str(sample["text"]))
        r2 = triage_once(str(sample["id"]) + "-repeat", str(sample["text"]))
        id1 = (r1.get("action_result") or {}).get("issue_id")
        id2 = (r2.get("action_result") or {}).get("issue_id")
        existed2 = (r2.get("action_result") or {}).get("existed")
        idem_ok = bool(id1 and id2 and id1 == id2 and existed2 is True)
        print("Idempotent OK:", idem_ok)


# --- FastAPI server (optional) ---
def serve(host: str = "127.0.0.1", port: int = 8000):
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="Support Triage Agent")

    class TriageIn(BaseModel):
        ticket_id: str
        text: str

    @app.post("/triage")
    def _triage(in_: TriageIn):
        out = triage_once(in_.ticket_id, in_.text)
        return out

    uvicorn.run(app, host=host, port=port)

    return app


# --- CLI ---
def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Support Triage Agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train classifier")
    p_train.add_argument(
        "--eval-split",
        type=float,
        default=0.0,
        help="Holdout ratio for quick eval (0.0 to disable)",
    )
    p_train.add_argument(
        "--clf-c", type=float, default=1.0, help="LogisticRegression C (regularization)"
    )

    p_index = sub.add_parser("index", help="Build FAISS index over history")

    p_triage = sub.add_parser("triage", help="Run triage on a single ticket")
    p_triage.add_argument("--ticket-id", required=True)
    p_triage.add_argument("--text", required=True)

    sub.add_parser("eval", help="Evaluate F1 + action + idempotency")

    p_serve = sub.add_parser("serve", help="Serve FastAPI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.cmd == "train":
        train_classifier(
            LABELED_CSV, CLF_PATH, eval_split=args.eval_split, clf_C=args.clf_c
        )
    elif args.cmd == "index":
        build_faiss(HISTORY_CSV)
    elif args.cmd == "triage":
        out = triage_once(args.ticket_id, args.text)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    elif args.cmd == "eval":
        evaluate()
    elif args.cmd == "serve":
        serve(args.host, args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
