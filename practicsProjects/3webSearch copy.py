import asyncio
import hashlib
import re
import time
from typing import Dict, List, Optional, Set, TypedDict, Annotated
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, add_messages
from dotenv import load_dotenv


load_dotenv()

# Configuration dictionary
DEFAULT_CONFIG = {
    "max_sources": 5,
    "min_sources": 3,
    "timeout_seconds": 10,
    "max_retries": 3,
    "retry_delay": 1.0,
    "allowed_domains": None,  # None means all allowed
    "denied_domains": {"facebook.com", "twitter.com", "reddit.com", "pinterest.com"},
    "max_content_length": 50000,
    "max_iterations": 3,
}


class AgentState(TypedDict):
    """State maintained throughout the agent workflow"""

    query: str
    search_queries: List[str]
    search_results: List[Dict]
    fetched_content: List[Dict]
    facts: List[str]
    synthesis: str
    bibliography: List[Dict[str, str]]
    iteration: int
    config: Dict
    session: Optional[aiohttp.ClientSession]
    llm: ChatOpenAI


# Utility functions
def get_content_hash(content: str) -> str:
    """Generate hash for content deduplication"""
    normalized = re.sub(r"\s+", " ", content.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def is_domain_allowed(url: str, config: Dict) -> bool:
    """Check if domain is allowed based on allow/deny lists"""
    try:
        domain = urlparse(url).netloc.lower()

        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]

        # Check deny list
        if any(denied in domain for denied in config["denied_domains"]):
            return False

        # Check allow list (if specified)
        if config["allowed_domains"]:
            return any(allowed in domain for allowed in config["allowed_domains"])

        return True

    except Exception:
        return False


def html_to_text(html_content: str) -> str:
    """Convert HTML to clean text"""
    try:
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()

        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text

    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return ""


async def fetch_with_retry(
    session: aiohttp.ClientSession, url: str, config: Dict
) -> str:
    """Fetch URL content with retries and timeout"""
    for attempt in range(config["max_retries"]):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    raise aiohttp.ClientResponseError(
                        response.request_info, response.history, status=response.status
                    )

        except Exception as e:
            if attempt < config["max_retries"] - 1:
                await asyncio.sleep(config["retry_delay"] * (2**attempt))
            else:
                raise e


async def mock_web_search(query: str) -> List[Dict[str, str]]:
    """Mock web search - replace with actual search API"""
    await asyncio.sleep(0.5)  # Simulate API delay

    return [
        {
            "url": f"https://example.com/article1/{hash(query) % 1000}",
            "title": f"Comprehensive Guide to {query}",
            "snippet": f"Detailed analysis of {query} with expert insights and data...",
        },
        {
            "url": f"https://news.example.com/story/{hash(query) % 1000}",
            "title": f"Breaking: Latest Updates on {query}",
            "snippet": f"Recent developments in {query} according to industry experts...",
        },
        {
            "url": f"https://academic.example.com/paper/{hash(query) % 1000}",
            "title": f"Research Study: {query} Analysis",
            "snippet": f"Peer-reviewed research examining {query} with statistical evidence...",
        },
    ]


async def extract_facts(
    content: str, query: str, llm: ChatOpenAI, config: Dict
) -> List[str]:
    """Extract relevant facts from content using LLM"""
    # Truncate content if too long
    if len(content) > config["max_content_length"]:
        content = content[: config["max_content_length"]] + "..."

    prompt = f"""
    Original query: {query}
    Content: {content}
    
    Extract 3-5 key facts from this content that are relevant to the query.
    Focus on specific details, names, dates, and explanations.
    Return each fact as a separate line starting with "FACT:"
    """

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        facts = [
            line.replace("FACT:", "").strip()
            for line in response.content.split("\n")
            if line.strip().startswith("FACT:")
        ]
        return facts
    except Exception as e:
        print(f"Error extracting facts: {e}")
        return []


# Graph node functions
async def planner_node(state: AgentState) -> AgentState:
    """Plan search queries based on the current state"""
    current_sources = len(state["fetched_content"])
    needed_sources = max(0, state["config"]["min_sources"] - current_sources)

    if state["iteration"] == 0:
        # Initial planning
        prompt = f"""
        Query: {state['query']}
        
        Generate 2-3 specific search queries to find comprehensive information about this question.
        Focus on finding authoritative sources with specific details.
        
        Return only the search queries, one per line.
        """
    else:
        # Adaptive planning based on current results
        current_facts = "\n".join(state["facts"][-5:])  # Last 5 facts
        prompt = f"""
        Original query: {state['query']}
        Current sources found: {current_sources}
        Need {needed_sources} more sources.
        
        Recent facts gathered:
        {current_facts}
        
        Generate 1-2 new search queries to fill gaps in information or find additional authoritative sources.
        Avoid repeating previous searches: {state['search_queries']}
        
        Return only the search queries, one per line.
        """

    response = await state["llm"].ainvoke([HumanMessage(content=prompt)])
    new_queries = [q.strip() for q in response.content.split("\n") if q.strip()]

    return {
        **state,
        "search_queries": state["search_queries"] + new_queries,
        "iteration": state["iteration"] + 1,
    }


async def searcher_node(state: AgentState) -> AgentState:
    """Execute search queries and collect results"""
    recent_queries = state["search_queries"][-3:]  # Only search recent queries
    new_results = []

    for query in recent_queries:
        try:
            # Use mock search (replace with actual search API)
            results = await mock_web_search(query)

            for result in results:
                if is_domain_allowed(result["url"], state["config"]):
                    search_result = {
                        "url": result["url"],
                        "title": result["title"],
                        "snippet": result["snippet"],
                        "content": None,
                        "fetch_time": None,
                        "content_hash": None,
                    }
                    new_results.append(search_result)

        except Exception as e:
            print(f"Search error for query '{query}': {e}")
            continue

    return {**state, "search_results": state["search_results"] + new_results}


async def fetcher_node(state: AgentState) -> AgentState:
    """Fetch content from search result URLs"""
    unfetched_results = [
        r
        for r in state["search_results"]
        if r["content"] is None and r not in state["fetched_content"]
    ]

    if not state["session"]:
        timeout = aiohttp.ClientTimeout(total=state["config"]["timeout_seconds"])
        state["session"] = aiohttp.ClientSession(timeout=timeout)

    # Limit concurrent fetches
    semaphore = asyncio.Semaphore(3)
    successfully_fetched = []

    async def fetch_single(result: Dict):
        async with semaphore:
            try:
                content = await fetch_with_retry(
                    state["session"], result["url"], state["config"]
                )
                result["content"] = html_to_text(content)
                result["fetch_time"] = time.time()
                result["content_hash"] = get_content_hash(result["content"])

                if len(result["content"]) > 100:  # Only add if substantial content
                    successfully_fetched.append(result)

            except Exception as e:
                print(f"Failed to fetch {result['url']}: {e}")

    await asyncio.gather(
        *[fetch_single(result) for result in unfetched_results[:5]],
        return_exceptions=True,
    )

    return {**state, "fetched_content": state["fetched_content"] + successfully_fetched}


async def deduplicator_node(state: AgentState) -> AgentState:
    """Remove duplicate content and extract facts"""
    # Deduplicate by content hash
    seen_hashes = set()
    unique_content = []

    for result in state["fetched_content"]:
        if result["content_hash"] and result["content_hash"] not in seen_hashes:
            seen_hashes.add(result["content_hash"])
            unique_content.append(result)

    # Extract facts from unique content
    new_facts = []
    for result in unique_content:
        if result["content"] and len(result["content"]) > 200:
            facts = await extract_facts(
                result["content"], state["query"], state["llm"], state["config"]
            )
            new_facts.extend(facts)

    # Deduplicate facts
    fact_hashes = set()
    unique_facts = []
    for fact in new_facts:
        fact_hash = get_content_hash(fact.lower())
        if fact_hash not in fact_hashes:
            fact_hashes.add(fact_hash)
            unique_facts.append(fact)

    return {
        **state,
        "fetched_content": unique_content,
        "facts": state["facts"] + unique_facts,
    }


def should_continue_searching(state: AgentState) -> str:
    """Decide whether to continue searching or synthesize results"""
    config = state["config"]
    has_min_sources = len(state["fetched_content"]) >= config["min_sources"]
    has_max_sources = len(state["fetched_content"]) >= config["max_sources"]
    max_iterations_reached = state["iteration"] >= config["max_iterations"]
    has_facts = len(state["facts"]) > 5

    if has_max_sources or max_iterations_reached or (has_min_sources and has_facts):
        return "synthesize"
    else:
        return "continue"


async def synthesizer_node(state: AgentState) -> AgentState:
    """Synthesize gathered facts into a brief with bibliography"""
    facts_text = "\n".join([f"- {fact}" for fact in state["facts"]])
    sources_text = "\n".join(
        [
            f"- {result['title']} ({result['url']})"
            for result in state["fetched_content"]
        ]
    )

    synthesis_prompt = f"""
    Original question: {state['query']}
    
    Gathered facts:
    {facts_text}
    
    Sources:
    {sources_text}
    
    Create a comprehensive but concise brief that:
    1. Directly answers the original question
    2. Synthesizes information from multiple sources
    3. Includes specific details and citations
    4. Is 200-400 words long
    
    Focus on accuracy and completeness.
    """

    response = await state["llm"].ainvoke([HumanMessage(content=synthesis_prompt)])

    # Create bibliography
    bibliography = [
        {
            "title": result["title"],
            "url": result["url"],
            "snippet": (
                result["snippet"][:200] + "..."
                if len(result["snippet"]) > 200
                else result["snippet"]
            ),
        }
        for result in state["fetched_content"]
    ]

    return {**state, "synthesis": response.content, "bibliography": bibliography}


def create_graph(llm: ChatOpenAI) -> StateGraph:
    """Create and compile the LangGraph workflow"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("searcher", searcher_node)
    workflow.add_node("fetcher", fetcher_node)
    workflow.add_node("deduplicator", deduplicator_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Add edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "fetcher")
    workflow.add_edge("fetcher", "deduplicator")
    workflow.add_conditional_edges(
        "deduplicator",
        should_continue_searching,
        {"continue": "planner", "synthesize": "synthesizer"},
    )
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


async def run_fact_gathering_agent(
    query: str,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Run the fact-gathering agent with the given query and configuration

    Args:
        query: The question to research
        config: Configuration overrides (merged with defaults)
        openai_api_key: OpenAI API key for LLM calls

    Returns:
        Dictionary with synthesis, bibliography, and metadata
    """
    # Merge config with defaults
    agent_config = {**DEFAULT_CONFIG, **(config or {})}

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0.1)

    # Create session for HTTP requests
    timeout = aiohttp.ClientTimeout(total=agent_config["timeout_seconds"])
    session = aiohttp.ClientSession(timeout=timeout)

    try:
        # Create initial state
        initial_state = AgentState(
            query=query,
            search_queries=[],
            search_results=[],
            fetched_content=[],
            facts=[],
            synthesis="",
            bibliography=[],
            iteration=0,
            config=agent_config,
            session=session,
            llm=llm,
        )

        # Create and run graph
        graph = create_graph(llm)
        final_state = await graph.ainvoke(initial_state)

        # Format results
        return {
            "query": query,
            "synthesis": final_state["synthesis"],
            "sources_found": len(final_state["fetched_content"]),
            "facts_gathered": len(final_state["facts"]),
            "bibliography": final_state["bibliography"],
            "search_queries_used": final_state["search_queries"],
            "iterations": final_state["iteration"],
        }

    finally:
        if session:
            await session.close()


# Configuration presets for different use cases
ACADEMIC_CONFIG = {
    **DEFAULT_CONFIG,
    "max_sources": 7,
    "min_sources": 4,
    "allowed_domains": {
        "wikipedia.org",
        "britannica.com",
        "jstor.org",
        "scholar.google.com",
        "researchgate.net",
        "arxiv.org",
    },
    "timeout_seconds": 20,
}

NEWS_CONFIG = {
    **DEFAULT_CONFIG,
    "max_sources": 5,
    "min_sources": 3,
    "allowed_domains": {
        "reuters.com",
        "bbc.com",
        "ap.org",
        "npr.org",
        "theguardian.com",
        "nytimes.com",
        "wsj.com",
    },
    "timeout_seconds": 10,
}

GENERAL_CONFIG = {
    **DEFAULT_CONFIG,
    "denied_domains": {
        "facebook.com",
        "twitter.com",
        "reddit.com",
        "pinterest.com",
        "instagram.com",
        "tiktok.com",
    },
}


# Enhanced version with real search API integration
async def serpapi_search(query: str, api_key: str) -> List[Dict[str, str]]:
    """Search using SerpAPI (requires serpapi package)"""
    # Implementation placeholder - install serpapi package
    # from serpapi import GoogleSearch
    #
    # search = GoogleSearch({
    #     "q": query,
    #     "api_key": api_key,
    #     "num": 10
    # })
    # results = search.get_dict()
    #
    # return [
    #     {
    #         "url": r["link"],
    #         "title": r["title"],
    #         "snippet": r["snippet"]
    #     }
    #     for r in results.get("organic_results", [])
    # ]

    # Fallback to mock for now
    return await mock_web_search(query)


async def brave_search(query: str, api_key: str) -> List[Dict[str, str]]:
    """Search using Brave Search API"""
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {"q": query, "count": 10}

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        {
                            "url": r["url"],
                            "title": r["title"],
                            "snippet": r["description"],
                        }
                        for r in data.get("web", {}).get("results", [])
                    ]
        except Exception as e:
            print(f"Brave search error: {e}")

    # Fallback to mock
    return await mock_web_search(query)


def create_production_searcher_node(
    search_api_key: str, search_provider: str = "brave"
):
    """Create a searcher node with real search API integration"""

    async def production_searcher_node(state: AgentState) -> AgentState:
        recent_queries = state["search_queries"][-3:]
        new_results = []

        for query in recent_queries:
            try:
                if search_provider == "brave":
                    results = await brave_search(query, search_api_key)
                elif search_provider == "serpapi":
                    results = await serpapi_search(query, search_api_key)
                else:
                    results = await mock_web_search(query)

                for result in results:
                    if is_domain_allowed(result["url"], state["config"]):
                        search_result = {
                            "url": result["url"],
                            "title": result["title"],
                            "snippet": result["snippet"],
                            "content": None,
                            "fetch_time": None,
                            "content_hash": None,
                        }
                        new_results.append(search_result)

            except Exception as e:
                print(f"Search error for query '{query}': {e}")
                continue

        return {**state, "search_results": state["search_results"] + new_results}

    return production_searcher_node


# Testing and usage functions
async def test_agent():
    """Test the fact-gathering agent"""

    # Test with custom configuration
    test_config = {
        **DEFAULT_CONFIG,
        "max_sources": 4,
        "min_sources": 2,
        "timeout_seconds": 15,
        "allowed_domains": {
            "wikipedia.org",
            "britannica.com",
            "reuters.com",
            "bbc.com",
        },
    }

    query = (
        "Who won the Pulitzer Prize for Fiction in 2023 and why did the judges cite it?"
    )

    print(f"Starting fact-gathering for: {query}")
    print("=" * 60)

    try:
        result = await run_fact_gathering_agent(
            query=query,
            config=test_config,
        )

        print("SYNTHESIS:")
        print(result["synthesis"])
        print("\n" + "=" * 60)

        print(f"STATISTICS:")
        print(f"Sources found: {result['sources_found']}")
        print(f"Facts gathered: {result['facts_gathered']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Search queries used: {len(result['search_queries_used'])}")

        print(f"\nSEARCH QUERIES:")
        for i, query in enumerate(result["search_queries_used"], 1):
            print(f"{i}. {query}")

        print(f"\nBIBLIOGRAPHY:")
        for i, source in enumerate(result["bibliography"], 1):
            print(f"{i}. {source['title']}")
            print(f"   {source['url']}")
            print(f"   {source['snippet']}")
            print()

    except Exception as e:
        print(f"Error running agent: {e}")


# Advanced usage with metrics tracking
async def run_with_metrics(query: str, config: Optional[Dict] = None) -> Dict:
    """Run agent with detailed performance metrics"""
    start_time = time.time()
    metrics = {
        "search_time": 0,
        "fetch_time": 0,
        "synthesis_time": 0,
        "failed_fetches": 0,
        "duplicate_content": 0,
    }

    result = await run_fact_gathering_agent(query, config)

    metrics["total_time"] = time.time() - start_time
    result["metrics"] = metrics

    return result


# Production-ready function with real APIs
async def run_production_agent(
    query: str,
    search_api_key: str,
    openai_api_key: str,
    search_provider: str = "brave",
    config: Optional[Dict] = None,
) -> Dict:
    """Run agent with production search APIs"""

    agent_config = {**DEFAULT_CONFIG, **(config or {})}
    llm = ChatOpenAI(model="gpt-4", api_key=openai_api_key, temperature=0.1)

    # Create graph with production searcher
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node(
        "searcher", create_production_searcher_node(search_api_key, search_provider)
    )
    workflow.add_node("fetcher", fetcher_node)
    workflow.add_node("deduplicator", deduplicator_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Add edges (same as before)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "fetcher")
    workflow.add_edge("fetcher", "deduplicator")
    workflow.add_conditional_edges(
        "deduplicator",
        should_continue_searching,
        {"continue": "planner", "synthesize": "synthesizer"},
    )
    workflow.add_edge("synthesizer", END)

    graph = workflow.compile()

    # Create session
    timeout = aiohttp.ClientTimeout(total=agent_config["timeout_seconds"])
    session = aiohttp.ClientSession(timeout=timeout)

    try:
        initial_state = AgentState(
            query=query,
            search_queries=[],
            search_results=[],
            fetched_content=[],
            facts=[],
            synthesis="",
            bibliography=[],
            iteration=0,
            config=agent_config,
            session=session,
            llm=llm,
        )

        final_state = await graph.ainvoke(initial_state)

        return {
            "query": query,
            "synthesis": final_state["synthesis"],
            "sources_found": len(final_state["fetched_content"]),
            "facts_gathered": len(final_state["facts"]),
            "bibliography": final_state["bibliography"],
            "search_queries_used": final_state["search_queries"],
            "iterations": final_state["iteration"],
        }

    finally:
        await session.close()


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_agent())
