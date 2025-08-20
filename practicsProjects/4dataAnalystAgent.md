# Data Analyst Agent: CSV-in, Question-out

Great brief. Here’s a lean, production-minded approach to a Data Analyst Agent that executes pandas safely in a sandbox and returns both the **answer** and the **code** it ran.

---

## Architecture (LangGraph)

### State

- **csv_bytes / csv_path** (single file for now)
- **schema** (column names + inferred dtypes)
- **question**
- **plan** (steps to compute)
- **code** (the pandas snippet to run)
- **exec_result** (answer, stdout/stderr, artifacts: e.g., dataframe head)
- **tests** (optional unit tests)
- **errors** (structured)

---

### Nodes

- **Schema Scanner** – load CSV (no exec), infer schema & sample
- **Planner** – turn question → calculation plan  
  _(e.g., "groupby category, sort by QoQ growth, take top 5")_
- **Codegen** – produce pure pandas + numpy code using a fixed template
- **Static Guard** – AST-lint the code  
  _(block imports other than allowed, no I/O, no attrs like `__subclasses__`, etc.)_
- **Sandbox Runner (Tool)** – run code in a resource-limited process; return answer + stdout + optional table preview
- **Validator** – (optional) run small unit tests against the sandbox result  
  _(e.g., expected columns, monotonicity)_
- **Synthesizer** – craft final answer + include the executed code
- **Controller** – if guard fails or tests fail → regenerate code (limited retries), else END

---

## Sandbox Design (safe-by-default)

Execution environment: separate subprocess (or container) with:

- **CPU time**: 5s  
  _(Linux `resource.setrlimit(RLIMIT_CPU, (5, 5))`)_
- **Memory cap**: e.g., 512 MB  
  _(RLIMIT_AS / cgroups)_
- **No network**: unset proxies + optionally drop network namespace/container or seccomp profile
- **Read-only FS**: copy CSV to a tmp dir; mount read-only or verify paths; no `open()`/write()
- **Whitelisting**: preload pandas, numpy only; no user imports (we provide them pre-imported)
- **Builtins**: restricted; remove `__import__`, `open`, `eval`, `exec`, `compile`, `input`, etc.
- **AST checks**: fail-fast on
  - `Import`
  - Attribute patterns like `__class__`, `__mro__`, `__subclasses__`, `os`, `sys`, `pathlib`, dunder names
  - Comprehensions with `__` names
- **Time/row guards**:
  - Pre-sample for planning but execute on full data
  - Cap rows returned _(e.g., `head(10)`) to bound memory in the response_

---

# Flow

1. **Schema Scanner**

   - Load CSV with safe options:
     ```python
     pd.read_csv(..., dtype_backend="pyarrow", on_bad_lines='skip', low_memory=False)
     ```
   - Infer schema = `[{"name": col, "dtype": str(df[col].dtype)}...]` and sample = `df.head(20)`.
   - Store only schema + a few rows in state (helps codegen).

2. **Planner (LLM)**

   - Prompt to produce semantic steps (`groupby` / `agg` / `sort` / `window`) and target shape.
   - Encourage explicit time logic (parse dates, sort by date, compute QoQ = `(current - prev)/prev`).

3. **Codegen (LLM)**

   - Emit code using a template (below) and column-safe helpers (e.g., `df.columns = df.columns.str.strip()`).

4. **Static Guard**

   - Parse AST and reject if:
     - **Nodes**: `Import*`, `With`, `Lambda`, `Global`, `Nonlocal`, `Raise`, `Try` (optional), attribute accessing forbidden names, calls to `__*__`.
     - **Identifier blacklist**: `os`, `sys`, `subprocess`, `builtins`, `pickle`, `exec`, `eval`, `open`, `compile`, `input`, `pathlib`, `importlib`, `signal`, `ctypes`, `resource`.
   - Enforce max lines (e.g., 120) and no `while` / `for` with huge ranges (limit range size).

5. **Sandbox Runner**

   - Spawn subprocess with resource caps; preload pandas as `pd`, numpy as `np`, load the full CSV into `df`.
   - Inject code into an `exec` with restricted globals.
   - Capture stdout/stderr. Enforce hard kill on timeout.
   - Return answer + up to N rows of any final DataFrame as preview.

6. **Validator (unit tests)**

   - Example: “top 5 categories by revenue growth QoQ”
   - **Pre-assert columns exist**: `['category','revenue','date']`
   - **After execution**:
     - Answer is list of `(category, growth)` sorted desc.
     - Length = 5.
     - Growth numeric.
     - Date treated monthly/quarterly.
     - No NaN.
   - Keep 10–20 lightweight tests; if any fail → Controller asks Codegen to refine (pass failing test messages back into prompt).

7. **Synthesizer**
   - Short natural-language answer (one paragraph),
   - Append executed code in a fenced block,
   - Optionally include a small table preview.
