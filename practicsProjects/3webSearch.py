from typing import TypedDict, Annotated, List, Dict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.types import RetryPolicy, CachePolicy
from tavily import TavilyClient
import httpx, asyncio, re, json
from urllib.parse import urlparse
import trafilatura


class RState(TypedDict, total=False):
    query: str
    plan: List[str]
    queries: List[str]
    hits: List[Dict]  # {title, url, snippet, score/vendor fields}
    pages: Dict[str, str]  # url -> html
    texts: Dict[str, str]  # url -> extracted text
    facts: List[Dict]  # normalized atomic facts
    sources: List[Dict]  # {url, title, publisher, date}
    errors: List[str]
    config: Dict  # {allow_domains, deny_domains, timeout_s, max_fetch, retries}
    messages: Annotated[List[BaseMessage], add_messages]  # optional for chat UX


llm = init_chat_model(model="openai:gpt-4o-mini")

tavily = TavilyClient(api_key="")  # reads TAVILY_API_KEY


# 1) Planner -> subquestions + diverse search queries
def planner(s: RState):
    prompt = f"""User question: {s['query']}
Return concise (1) sub-questions and (2) 4–7 web search queries that include synonyms, year, and site filters if helpful (e.g., site:.org, site:.edu).
Respond as JSON: {{"plan": [...], "queries": [...]}}"""
    out = llm.invoke([HumanMessage(content=prompt)]).content
    j = json.loads(out)
    return {"plan": j.get("plan", []), "queries": j.get("queries", [])}


# 2) Searcher (Tavily)
def searcher(s: RState):
    hits = []
    for q in s.get("queries", []):
        # Tavily search; tune max_results/search_depth to your budget
        res = tavily.search(q, max_results=8, search_depth="advanced")
        # normalize to {title, url, snippet}
        for item in res.get("results", []):
            hits.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("content") or item.get("snippet", ""),
                }
            )
    return {"hits": hits}


# 3) Gatekeeper: allow/deny + dedupe title/domain
def gatekeeper(s: RState):
    allowed = set((s.get("config", {}).get("allow_domains") or []))
    denied = set((s.get("config", {}).get("deny_domains") or []))
    keep, seen = [], set()
    for h in s.get("hits", []):
        host = urlparse(h["url"]).netloc.replace("www.", "")
        if (allowed and host not in allowed) or host in denied:
            continue
        key = f"{host}:{re.sub(r'\\W+',' ', (h['title'] or '').lower())[:120]}"
        if key in seen:
            continue
        seen.add(key)
        keep.append(h)
    return {"hits": keep}


# 4) Fetcher: httpx with timeout & retries
async def _fetch_one(url, timeout_s):
    async with httpx.AsyncClient(
        timeout=timeout_s,
        follow_redirects=True,
        headers={"User-Agent": "ResearchAgent/1.0 (+langgraph)"},
    ) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.text


def fetcher(s: RState):
    pages = {}
    timeout_s = s.get("config", {}).get("timeout_s", 10)
    max_fetch = s.get("config", {}).get("max_fetch", 8)
    for h in s.get("hits", [])[:max_fetch]:
        try:
            html = asyncio.run(_fetch_one(h["url"], timeout_s))
            pages[h["url"]] = html
        except Exception as e:
            s.setdefault("errors", []).append(f"fetch {h['url']}: {e}")
    return {"pages": pages}


# 5) Extractor: Trafilatura main content
def extractor(s: RState):
    texts = {}
    for url, html in (s.get("pages") or {}).items():
        txt = trafilatura.extract(html, url=url, favor_precision=True) or ""
        texts[url] = txt.strip()
    return {"texts": texts}


# 6) Fact miner: normalize to award/year/winner/citation
def fact_miner(s: RState):
    items = []
    for url, text in (s.get("texts") or {}).items():
        if not text:
            continue
        schema = {
            "type": "object",
            "properties": {
                "award": {"type": "string"},
                "year": {"type": "string"},
                "category": {"type": "string"},
                "winner": {"type": "string"},
                "judges_citation": {"type": "string"},
                "supporting_quote": {"type": "string"},
                "publisher": {"type": "string"},
                "pub_date": {"type": "string"},
            },
            "required": ["winner"],
        }
        msg = f"""From the text, extract fields. If unsure leave blank. 
Return JSON only.
URL: {url}
TEXT:
{text[:8000]}"""
        out = llm.with_structured_output(schema).invoke([HumanMessage(content=msg)])
        d = dict(out)
        d["url"] = url
        items.append(d)
    return {"facts": items}


# 7) Synthesizer: brief + bib
def synthesizer(s: RState):
    facts = s.get("facts", [])
    # de-dupe by (award, year, category, winner)
    seen, keep = set(), []
    for f in facts:
        k = (
            f.get("award", "").lower(),
            f.get("year", ""),
            f.get("category", "").lower(),
            f.get("winner", "").lower(),
        )
        if k in seen:
            continue
        seen.add(k)
        keep.append(f)

    prompt = f"""Write a concise brief (<=8 sentences) answering:
{ s['query'] }

Use only these normalized facts; highlight disagreements if any. Then output a short bibliography (3–6 items).
FACTS (JSON):
{json.dumps(keep, ensure_ascii=False)}
"""
    brief = llm.invoke([HumanMessage(content=prompt)]).content

    # simple bib (you can enrich with titles/publishers found earlier)
    bib = [
        {
            "url": f.get("url"),
            "publisher": f.get("publisher"),
            "date": f.get("pub_date"),
        }
        for f in keep
        if f.get("url")
    ]
    return {"messages": [HumanMessage(content=brief)], "sources": bib}


graph = StateGraph(RState)
graph.add_node("planner", planner)
graph.add_node("searcher", searcher)
graph.add_node("gatekeeper", gatekeeper)
graph.add_node("fetcher", fetcher)
graph.add_node("extractor", extractor)
graph.add_node("fact_miner", fact_miner)
graph.add_node("synthesizer", synthesizer)

graph.add_edge(START, "planner")
graph.add_edge("planner", "searcher")
graph.add_edge("searcher", "gatekeeper")
graph.add_edge("gatekeeper", "fetcher")
graph.add_edge("fetcher", "extractor")
graph.add_edge("extractor", "fact_miner")


def need_more(s: RState):
    facts = s.get("facts", [])
    uniq_hosts = {urlparse(f.get("url", "")).netloc for f in facts if f.get("url")}
    has_citation = any((f.get("judges_citation") or "").strip() for f in facts)
    return "planner" if (len(uniq_hosts) < 3 or not has_citation) else "synthesizer"


graph.add_conditional_edges(
    "fact_miner", need_more, {"planner": "planner", "synthesizer": "synthesizer"}
)
graph.add_edge("synthesizer", END)

# Compile with retries/caching; also you can set per-node policies.
compiled = graph.compile(
    retry_policy=RetryPolicy(max_attempts=3, backoff_factor=2.0),  # network resilience
    cache_policy=CachePolicy(),  # cache pure nodes like fetch/extract
)  # Policies documented in LangGraph reference.  [oai_citation:5‡Langchain AI](https://langchain-ai.github.io/langgraph/reference/graphs/?utm_source=chatgpt.com)


cfg = {
    "configurable": {
        "thread_id": "research-1"
    },  # optional if you want checkpoints/memory
}
state = {
    "query": "Who won the X Award in 2021 and what did the judges cite?",
    "config": {
        "allow_domains": [],  # e.g., ["reuters.com","bbc.com","nytimes.com"]
        "deny_domains": ["reddit.com", "facebook.com", "x.com"],
        "timeout_s": 10,
        "max_fetch": 8,
    },
}
result = compiled.invoke(state, config=cfg)
brief = result["messages"][-1].content
bib = result.get("sources", [])
print(brief)
print(bib)
