# architecture at a glance

## State (TypedDict)

The state tracks the entire pipeline:

- **query**
- **messages**
- **plan**
- **queries**
- **hits**
- **pages**
- **facts**
- **sources**
- **errors**
- **config** (timeouts, retries, allow/deny domains)

> Use `add_messages` reducer for handling messages.
> • Nodes 1. Planner – turn the user question into sub-questions + search queries 2. Searcher – web search API → list of candidate URLs 3. Gatekeeper – apply allow/deny list, dedupe domains 4. Fetcher – HTTP GET with timeout/retries; store HTML and HTTP metadata 5. Extractor – HTML→text (normalize, strip boilerplate) 6. Fact Miner – pull structured facts (winner, year, judges’ citation, quotes) 7. Synthesizer – write a concise brief with inline source refs; build bibliography 8. Controller – loop Planner ↔ Searcher until coverage ≥ N sources or budget hit; then exit
> (If you prefer prebuilts for tool routing, pair ToolNode with tools_condition.) ￼
> • Control
> Use conditional edges for the “keep searching?” loop, retries per node, and optional cache policy for fetch/parse steps.

## nodes (how each one works)

### 1. Planner

- Converts the user question into **2–5 sub-questions** and search queries.
- Queries should include:
  - Name variations
  - `site:.org` refinements, etc.
- Returns:
  - **plan**, **queries**
- Note: The **graph API** composes sequences/branches cleanly.

---

### 2. Searcher

- Calls a search API tool (e.g., **SerpAPI, Tavily, Bing**).
- For each `queries[i]`, collects top _k_ results → **hits**.
- Retry policy: e.g., exponential backoff on HTTP 429/5xx.

---

### 3. Gatekeeper

- Applies **allow_domains / deny_domains** rules.
- Drops non-HTML results.
- Deduplicates by **domain + title shingle**.
- Ensures diversity: news, official sites, encyclopedias.

---

### 4. Fetcher

- Performs **HTTP GET** with:
  - Timeout (e.g., 10s)
  - Retries
- Persists response text in `pages[url]`.
- Can optionally be **cached**.

---

### 5. Extractor

- Converts **HTML → text**.
- Uses async loaders + `html2text` / Readability-like boilerplate stripping.
- Stores text in `texts[url]`.

---

### 6. Fact Miner

- Prompts model to extract structured fields:
- Appends extracted information to **facts**.
- If fields are missing (e.g., year/citation), mark **confidence: low**.

---

### 7. Synthesizer

- Aggregates and normalizes across **facts**.
- De-duplicates by normalized key: _(award, year, category, winner)_.
- Produces:
  - A **concise brief** (5–8 sentences).
  - A **short bibliography**: title, publisher, date, URL.
- Notes disagreements when sources conflict.

---

### 8. Controller (Loop)

- Checks conditions:
  - If `len(unique_sources) < 3` **OR** `judges_citation` missing → branch back to **Planner/Searcher** with refined queries.
  - Else → **END**.
- Uses **conditional edges** for routing.
- Alternatively, pair with **prebuilt agent helpers** when binding tools.

---

### Reliability & Governance

- **Timeouts & Retries**

  - Set `retry_policy` per node.
  - Respect `timeout_s` in Fetcher.
  - Differentiate between rate-limit errors and empty-body responses.
  - For complex cases, use a dedicated **error handler** node.

- **Caching**

  - Attach `CachePolicy` to Fetcher/Extractor.
  - Avoids refetching identical URLs.

- **Allow/Deny Lists**

  - Enforced in Gatekeeper.
  - Re-applied in Synthesizer when selecting citations.

- **HIL (Human-in-the-Loop, optional)**

  - `interrupt_before=["synthesizer"]` allows review before publishing.

- **Observability**
  - Run in **Studio/Server**.
  - Trace via **LangSmith** during development.

---
