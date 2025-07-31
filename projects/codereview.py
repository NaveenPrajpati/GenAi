from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import TypedDict, Annotated

load_dotenv()


prompt = PromptTemplate.from_template(
    """
You are a senior software engineer performing a code review.
Step through the following code in detail.
For each relevant part:
1. Explain what it does.
2. Identify potential bugs or issues.
3. Provide suggestions for improvement.
4. Summarize all findings concisely.
Code:
{code}
""",
)


# Pydantic Steps
class CodeStepPydantic(BaseModel):
    step_number: int = Field(..., description="Step order in the review")
    description: str = Field(..., description="What this section/code does")
    issue: Optional[str] = Field(
        None, description="Potential bug or code smell detected here"
    )
    reason: Optional[str] = Field(
        None, description="Why the suggestion or issue matters"
    )
    suggestion: Optional[str] = Field(
        None, description="Suggested improvement for this step"
    )


# TypedDict Steps
class CodeStepTypedDict(TypedDict):
    step_number: Annotated[int, ..., "Step order in the review"]
    description: Annotated[str, ..., "What this section/code does"]
    issue: Annotated[Optional[str], None, "Potential bug or code smell detected here"]
    reason: Annotated[Optional[str], None, "Why the suggestion or issue matters"]
    suggestion: Annotated[Optional[str], None, "Suggested improvement for this step"]


# Pydantic output
class CodeReviewReportPydantic(BaseModel):
    filename: Optional[str] = Field(None, description="Name of the file reviewed")
    summary: str = Field(..., description="Overall summary of the code review")
    steps: List[CodeStepPydantic] = Field(
        ..., description="Step-by-step breakdown of findings"
    )


# TypedDict
class CodeReviewReportTypedDict(TypedDict):
    """Code Review"""

    summary: Annotated[str, ..., "Overall summary of the code review"]
    filename: Annotated[Optional[str], None, "Name of the file reviewed"]
    steps: Annotated[
        List[CodeStepTypedDict], ..., "Step-by-step breakdown] of findings"
    ]

    # Alternatively, we could have specified setup as:

    # setup: str                    # no default, no description
    # setup: Annotated[str, ...]    # no default, no description
    # setup: Annotated[str, "foo"]  # default, no description


# json schema dict similar output as TypeDict
json_schem = {
    "title": "CodeReviewReport",
    "type": "object",
    "description": "Code Review",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Overall summary of the code review",
        },
        "filename": {
            "type": ["string", "null"],
            "description": "Name of the file reviewed",
        },
        "steps": {
            "type": "array",
            "description": "Step-by-step breakdown of findings",
            "items": {
                "type": "object",
                "properties": {
                    "step_number": {
                        "type": "integer",
                        "description": "Step order in the review",
                    },
                    "description": {
                        "type": "string",
                        "description": "What this section/code does",
                    },
                    "issue": {
                        "type": ["string", "null"],
                        "description": "Potential bug or code smell detected here",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "Why the suggestion or issue matters",
                    },
                    "suggestion": {
                        "type": ["string", "null"],
                        "description": "Suggested improvement for this step",
                    },
                },
                "required": ["step_number", "description"],
            },
        },
    },
    "required": ["summary", "steps"],
}


llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(json_schem)

chain = prompt | llm

result = chain.invoke({"code": "function foo(x){ return x+1/0}"})

# streaming is support for TypedDict class or JSON Schema dict
# for result in chain.stream({"code": "def foo(x): return x+1/0"}):
print(result)


COT_PROMPT = """
You are a senior code reviewer. Review the following code step by step:

**Step 1: Code Understanding**
First, analyze what this code does:
- Purpose and functionality
- Key components and flow
- Technologies/frameworks used

**Step 2: Issue Detection**
Look for these categories of issues:
- Bugs and logical errors
- Security vulnerabilities
- Performance problems
- Code smells

**Step 3: Best Practices Check**
Evaluate against:
- Clean code principles
- Language-specific conventions
- Architecture patterns
- Documentation quality

**Step 4: Recommendations**
Provide specific, actionable improvements with:
- Priority level (High/Medium/Low)
- Code examples where helpful
- Reasoning for each suggestion

Code to review:
{code}

Please work through each step systematically.
"""
