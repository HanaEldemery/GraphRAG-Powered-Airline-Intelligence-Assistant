from openai import OpenAI
import os
import time
import json
import re
from typing import Any, Dict, Optional, Tuple, Union

def _get_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY (or OPENAI_API_KEY).")
    return OpenAI(
        base_url="",
        api_key=api_key,
    )


def _normalize_usage(usage: Any) -> Dict[str, Any]:
    if not usage:
        return {"prompt": None, "completion": None, "total": None}

    # OpenAI SDK usage object
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
        return {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens}

    # Dict-like usage (common for OpenRouter)
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        total_tokens = usage.get("total_tokens")
        if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
            try:
                total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
            except Exception:
                total_tokens = None
        return {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens}

    return {"prompt": None, "completion": None, "total": None}


def generate_answer(
    prompt: str,
    context: str,
    model: str,
    *,
    return_metrics: bool = False,
) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """Generate an answer to the prompt using the provided context and model."""
    client = _get_openrouter_client()
    start = time.perf_counter()
    
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": f"""You are an expert airline flight information assistant helping customers understand flight data.

CONTEXT DATA:
{context}

CRITICAL INSTRUCTIONS:
1. NEVER mention or include database IDs, node IDs, or technical identifiers in your response
2. NEVER mention labels like 'Airport', 'Flight', 'Journey', 'Passenger' - use natural language instead
3. NEVER mention relationship types like 'DEPARTS_FROM', 'ARRIVES_AT', 'ON', 'TOOK' - describe them naturally
4. Focus ONLY on flight-relevant information: flight numbers, airport codes, aircraft types, routes, satisfaction scores, delays, passenger classes, loyalty levels
5. If the data is empty or insufficient, clearly state that you cannot find the requested information
6. Provide concise, customer-friendly answers
7. Use natural aviation terminology (e.g., "departs from", "arrives at", "operates on", "travels from X to Y")

RESPONSE FORMAT:
- Present information in clear, complete sentences
- Use bullet points ONLY when listing multiple items
- Include specific details: flight numbers, airport codes, aircraft types, scores, delays
- Be conversational and helpful, not technical

EXAMPLE TRANSFORMATIONS:
❌ Bad: "Node with label 'Flight' has property flight_number: 2411"
✅ Good: "Flight 2411 operates this route"

❌ Bad: "Relationship DEPARTS_FROM connects to node with station_code LAX"
✅ Good: "The flight departs from LAX"

❌ Bad: "Journey node with food_satisfaction_score property"
✅ Good: "Passengers rated the food 3.5 out of 5"

Now answer the user's question using ONLY the provided context."""
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    answer = completion.choices[0].message.content
    if not return_metrics:
        return answer

    metrics = {
        "response_time_ms": elapsed_ms,
        "tokens": _normalize_usage(getattr(completion, "usage", None)),
    }
    return answer, metrics


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def judge_answer(
    *,
    question: str,
    context: str,
    answer: str,
    cypher_query: Optional[str] = None,
    model: str,
    return_metrics: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    LLM-as-judge qualitative evaluation for a generated answer.

    Returns a qualitative rubric evaluation (no numeric scores).
    """
    client = _get_openrouter_client()

    judge_payload = {
        "question": question,
        "context": context,
        "cypher_query": cypher_query,
        "answer": answer,
    }

    system = (
        "You are a strict evaluator for a QA system.\n"
        "Evaluate the answer using ONLY the provided context.\n"
        "If the answer includes claims not supported by the context, mark it as weak groundedness and call out the unsupported claims.\n\n"
        "Do NOT output numeric scores.\n"
        "Use these labels exactly: Excellent, Good, Mixed, Poor.\n"
        "Return ONLY valid JSON with this schema (no extra keys):\n"
        "{\n"
        '  "overall_verdict": "Strong|Acceptable|Weak",\n'
        '  "dimension_feedback": [\n'
        "    {\n"
        '      "dimension": "Groundedness|Completeness|Clarity|ConstraintFollowing",\n'
        '      "label": "Excellent|Good|Mixed|Poor",\n'
        '      "evidence": ["1-3 short quotes/snippets from the answer or context"],\n'
        '      "commentary": "1-3 short sentences"\n'
        "    }\n"
        "  ],\n"
        '  "top_strengths": ["1-3 short bullets"],\n'
        '  "top_improvements": ["1-3 short bullets"],\n'
        '  "rewrite_suggestion": "1-3 sentences suggesting how to improve (no full rewrite)"\n'
        "}\n"
        "No markdown. No code fences. Ensure JSON is parseable."
    )

    start = time.perf_counter()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(judge_payload, ensure_ascii=False)},
        ],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    raw = completion.choices[0].message.content or ""
    parsed = _extract_json_object(raw) or {"error": "Judge did not return valid JSON.", "raw": raw}

    if not return_metrics:
        return parsed

    metrics = {
        "response_time_ms": elapsed_ms,
        "tokens": _normalize_usage(getattr(completion, "usage", None)),
    }
    return parsed, metrics

if __name__ == "__main__":
    prompt = "What is the status of flight AA123?"
    context = "Flight AA123 is scheduled to depart at 3:00 PM and arrive at 6:00 PM. It is currently on time."
    model = "meta-llama/llama-3.3-70b-instruct:free"

    answer = generate_answer(prompt, context, model)
    print("Answer:", answer)
