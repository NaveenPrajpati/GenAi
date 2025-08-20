from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
import ast, json, os, signal, sys, tempfile, textwrap, time
import resource, subprocess, shlex, pathlib, re


#  Sandbox runner (subprocess; Linux/macOS)
ALLOWED_BUILTINS = {
    "len",
    "range",
    "min",
    "max",
    "sum",
    "sorted",
    "enumerate",
    "list",
    "dict",
    "set",
    "tuple",
    "abs",
    "round",
    "all",
    "any",
    "zip",
    "str",
    "int",
    "float",
    "bool",
    "type",
    "isinstance",
    "hasattr",
}

FORBIDDEN_NAMES = {
    "os",
    "sys",
    "subprocess",
    "builtins",
    "pickle",
    "eval",
    "exec",
    "open",
    "compile",
    "input",
    "pathlib",
    "importlib",
    "ctypes",
    "signal",
    "resource",
    "globals",
    "locals",
    "vars",
    "dir",
}

# Common analytical patterns for better code generation
ANALYSIS_PATTERNS = {
    "top_n": r"top\s+(\d+)|first\s+(\d+)|highest\s+(\d+)",
    "bottom_n": r"bottom\s+(\d+)|last\s+(\d+)|lowest\s+(\d+)",
    "growth": r"growth|increase|decrease|change|trend",
    "quarter": r"q[1-4]|quarter|quarterly|qoq",
    "comparison": r"vs|versus|compare|compared|against",
    "aggregation": r"total|sum|average|mean|count|max|min",
}


def ast_guard(src: str):
    """Enhanced AST validation with better error messages"""
    try:
        tree = ast.parse(src, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}")

    for node in ast.walk(tree):
        # Check forbidden node types
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements not allowed")
        if isinstance(node, (ast.With, ast.Lambda)):
            raise ValueError(f"{type(node).__name__} not allowed")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("Global/nonlocal statements not allowed")

        # Check attribute access
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise ValueError(f"Dunder attribute access not allowed: {node.attr}")

        # Check function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
                raise ValueError(f"Forbidden function call: {node.func.id}")

        # Check name usage
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                raise ValueError(f"Forbidden name: {node.id}")
            if node.id.startswith("__") and node.id.endswith("__"):
                raise ValueError(f"Dunder name not allowed: {node.id}")

    return True


def extract_code_from_llm_response(response: str) -> str:
    """Extract Python code from LLM response, handling markdown formatting"""
    # Remove markdown code blocks
    code = re.sub(r"```python\s*\n?", "", response)
    code = re.sub(r"```\s*$", "", code)

    # Remove common LLM prefixes
    lines = code.split("\n")
    cleaned_lines = []
    for line in lines:
        # Skip explanation lines
        if line.strip().startswith("#") and any(
            word in line.lower() for word in ["here", "this", "code", "will"]
        ):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def run_in_sandbox(
    csv_path: str, user_code: str, timeout_s: int = 5, mem_mb: int = 512
) -> dict:
    """Enhanced sandbox with better error handling"""
    try:
        ast_guard(user_code)
    except ValueError as e:
        return {"error": "validation", "message": str(e)}

    # Enhanced driver with better error capture
    driver = f"""
import json, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    df = pd.read_csv({csv_path!r}, low_memory=False)
    answer = None
    
    # ---- USER CODE START ----
{textwrap.indent(user_code, '    ')}
    # ---- USER CODE END ----
    
    # Ensure answer is JSON serializable
    if hasattr(answer, 'to_dict'):
        answer = answer.to_dict()
    elif hasattr(answer, 'tolist'):
        answer = answer.tolist()
    elif hasattr(answer, 'item'):
        answer = answer.item()
        
    result = {{"answer": answer, "success": True}}
    
except Exception as e:
    result = {{"error": str(e), "success": False, "error_type": type(e).__name__}}

print(json.dumps(result, ensure_ascii=False, default=str))
"""

    with tempfile.TemporaryDirectory() as td:
        runner_path = os.path.join(td, "runner.py")
        with open(runner_path, "w") as f:
            f.write(driver)

        def preexec():
            # Resource limits
            resource.setrlimit(resource.RLIMIT_CPU, (timeout_s, timeout_s))
            bytes_cap = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (bytes_cap, bytes_cap))
            resource.setrlimit(
                resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024)
            )
            os.setsid()

        p = subprocess.Popen(
            [sys.executable, "-I", runner_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=preexec,
            env={"PYTHONWARNINGS": "ignore"},
            cwd=td,
        )

        try:
            out, err = p.communicate(timeout=timeout_s + 2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except:
                pass
            return {"error": "timeout", "message": "Code execution timed out"}

    if p.returncode != 0:
        return {
            "error": "runtime",
            "message": err.decode("utf-8", "ignore")[:500],
            "stdout": out.decode("utf-8", "ignore")[:500],
        }

    try:
        result = json.loads(out.decode("utf-8", "ignore"))
        return result
    except json.JSONDecodeError as e:
        return {
            "error": "json_decode",
            "message": f"Failed to parse output: {str(e)}",
            "raw_output": out.decode("utf-8", "ignore")[:200],
        }


# Enhanced graph nodes
llm = init_chat_model(model="openai:gpt-4o-mini")


def schema_scanner(state):
    """Enhanced schema analysis with data profiling"""
    import pandas as pd
    import io

    try:
        df = pd.read_csv(io.BytesIO(state["csv_bytes"]), low_memory=False)

        schema = []
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique()),
            }

            # Add sample values for categorical columns
            if df[col].dtype == "object" or df[col].nunique() < 20:
                col_info["sample_values"] = df[col].value_counts().head(5).to_dict()

            schema.append(col_info)

        # Smart sampling - include edge cases
        sample_data = []
        if len(df) > 0:
            # First few rows
            sample_data.extend(df.head(3).to_dict(orient="records"))
            # Last few rows if different
            if len(df) > 6:
                sample_data.extend(df.tail(2).to_dict(orient="records"))

        return {
            "schema": schema,
            "sample": sample_data,
            "row_count": len(df),
            "column_count": len(df.columns),
        }
    except Exception as e:
        return {"error": f"Schema scan failed: {str(e)}"}


def planner(state):
    """Enhanced planning with pattern recognition"""
    question_lower = state["question"].lower()

    # Detect analytical patterns
    detected_patterns = []
    for pattern_name, regex in ANALYSIS_PATTERNS.items():
        if re.search(regex, question_lower):
            detected_patterns.append(pattern_name)

    prompt = f"""Analyze this data question and create a step-by-step plan:

Question: {state['question']}
Dataset info: {state['row_count']} rows, {state['column_count']} columns

Schema (with sample values):
{json.dumps(state['schema'][:5], indent=2)}

Detected patterns: {detected_patterns}

Create a clear, specific plan using pandas operations. Focus on:
1. What columns to use
2. Any filtering needed
3. Grouping/aggregation steps
4. Sorting/ranking requirements
5. Final output format

Plan:"""

    try:
        plan = llm.invoke([HumanMessage(content=prompt)]).content
        return {"plan": plan, "detected_patterns": detected_patterns}
    except Exception as e:
        return {"error": f"Planning failed: {str(e)}"}


def codegen(state):
    """Enhanced code generation with examples and validation"""
    retry_count = state.get("retry_count", 0)

    # Include previous errors for better retry attempts
    error_context = ""
    if retry_count > 0 and "previous_errors" in state:
        error_context = f"\nPrevious attempt failed with: {state['previous_errors'][-1]}\nPlease fix these issues."

    prompt = f"""Write pandas code to answer: "{state['question']}"

Dataset Schema:
{json.dumps(state['schema'][:8], indent=2)}

Analysis Plan:
{state.get('plan', 'No plan available')}

Requirements:
- Use only pandas (pd) and numpy (np) - they're already imported
- DataFrame is available as 'df'
- End with: answer = <your_result>
- Keep answer small (top 10 results max)
- Handle missing values appropriately
- Use appropriate data types

Code format:
```python
# Your pandas code here
answer = final_result
```

{error_context}

Generate clean, efficient pandas code:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content
        code = extract_code_from_llm_response(response)

        return {"code": code, "retry_count": retry_count}
    except Exception as e:
        return {"error": f"Code generation failed: {str(e)}"}


def static_guard(state):
    """Enhanced static analysis"""
    errors = []

    try:
        ast_guard(state["code"])
    except Exception as e:
        errors.append(f"AST validation: {str(e)}")

    # Additional checks
    code_lines = state["code"].lower()
    if "import" in code_lines and "pd" not in code_lines and "np" not in code_lines:
        errors.append("Unexpected import statement detected")

    if not re.search(r"answer\s*=", state["code"]):
        errors.append("Code must end with 'answer = <result>'")

    return {"validation_errors": errors} if errors else {}


def exec_node(state):
    """Enhanced execution with better error handling"""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(state["csv_bytes"])
        csv_path = f.name

    try:
        result = run_in_sandbox(csv_path, state["code"], timeout_s=5, mem_mb=512)
        return {"exec_result": result}
    except Exception as e:
        return {"exec_result": {"error": "execution", "message": str(e)}}
    finally:
        try:
            os.remove(csv_path)
        except:
            pass


def validator(state):
    """Enhanced validation with specific checks"""
    errors = []
    result = state["exec_result"]

    # Check execution success
    if not result.get("success", False) or "error" in result:
        error_msg = result.get("message", result.get("error", "Unknown error"))
        errors.append(f"Execution failed: {error_msg}")

    # Validate answer format
    answer = result.get("answer")
    if answer is None:
        errors.append("No answer returned")
    elif isinstance(answer, (list, dict)):
        if len(str(answer)) > 2000:  # Prevent huge outputs
            errors.append("Answer too large, please limit results")

    # Track errors for retry logic
    previous_errors = state.get("previous_errors", [])
    if errors:
        previous_errors.append(errors[0])  # Keep track for better retries

    return {
        "validation_errors": errors,
        "previous_errors": previous_errors,
        "retry_count": state.get("retry_count", 0) + (1 if errors else 0),
    }


def synthesizer(state):
    """Final response synthesis"""
    validation_errors = state.get("validation_errors", [])

    if validation_errors:
        return {
            "final": {
                "success": False,
                "errors": validation_errors,
                "code": state.get("code", ""),
                "retry_count": state.get("retry_count", 0),
            }
        }

    return {
        "final": {
            "success": True,
            "answer": state["exec_result"]["answer"],
            "code": state["code"],
            "question": state["question"],
        }
    }


# Build the graph with retry limits
def create_graph():
    g = StateGraph(dict)

    g.add_node("schema", schema_scanner)
    g.add_node("planner", planner)
    g.add_node("codegen", codegen)
    g.add_node("guard", static_guard)
    g.add_node("exec", exec_node)
    g.add_node("validate", validator)
    g.add_node("synthesize", synthesizer)

    # Linear flow with conditional retry logic
    g.add_edge(START, "schema")
    g.add_edge("schema", "planner")
    g.add_edge("planner", "codegen")
    g.add_edge("codegen", "guard")

    def route_after_guard(state):
        max_retries = 3
        has_errors = bool(state.get("validation_errors"))
        retry_count = state.get("retry_count", 0)

        if has_errors and retry_count < max_retries:
            return "codegen"  # Retry code generation
        return "exec"

    g.add_conditional_edges(
        "guard", route_after_guard, {"codegen": "codegen", "exec": "exec"}
    )
    g.add_edge("exec", "validate")

    def route_after_validate(state):
        max_retries = 3
        has_errors = bool(state.get("validation_errors"))
        retry_count = state.get("retry_count", 0)

        if has_errors and retry_count < max_retries:
            return "codegen"  # Retry from code generation
        return "synthesize"

    g.add_conditional_edges(
        "validate",
        route_after_validate,
        {"codegen": "codegen", "synthesize": "synthesize"},
    )
    g.add_edge("synthesize", END)

    return g.compile()


# Usage example
graph = create_graph()


def analyze_csv(csv_bytes: bytes, question: str) -> dict:
    """Main entry point for CSV analysis"""
    initial_state = {
        "csv_bytes": csv_bytes,
        "question": question,
        "retry_count": 0,
        "previous_errors": [],
    }

    try:
        result = graph.invoke(initial_state)
        return result["final"]
    except Exception as e:
        return {
            "success": False,
            "errors": [f"Graph execution failed: {str(e)}"],
            "code": "",
            "question": question,
        }
