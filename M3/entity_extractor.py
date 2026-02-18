import re
import os
import json
import requests
import time
from typing import Dict, List, Optional

# LLM Configuration (prefer env vars)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Valid entities from the dataset (used by both regex and LLM extraction)
VALID_AIRPORT_CODES = [
    'ABX', 'ACX', 'ALX', 'AMX', 'ANX', 'ASX', 'ATX', 'AUX', 'BCX', 'BDX', 'BEX', 'BGX', 'BHX', 'BIX',
    'BJX', 'BNX', 'BOX', 'BQX', 'BRX', 'BTX', 'BUX', 'BWX', 'BZX', 'CDX', 'CHX', 'CIX', 'CLX', 'CMX',
    'COX', 'CPX', 'CRX', 'CUX', 'CVX', 'CZX', 'DBX', 'DCX', 'DEX', 'DFX', 'DSX', 'DTX', 'DUX', 'EDX',
    'ELX', 'EUX', 'EWX', 'EZX', 'FAX', 'FCX', 'FLX', 'FRX', 'FSX', 'GDX', 'GEX', 'GIX', 'GJX', 'GRX',
    'GSX', 'GTX', 'GUX', 'GVX', 'HDX', 'HNX', 'HSX', 'IAX', 'ICX', 'IDX', 'INX', 'ITX', 'JAX', 'JFX',
    'JNX', 'KEX', 'KOX', 'LAX', 'LGX', 'LHX', 'LIX', 'LNX', 'LOX', 'MAX', 'MBX', 'MCX', 'MDX', 'MEX',
    'MFX', 'MIX', 'MKX', 'MNX', 'MSX', 'MTX', 'MUX', 'MXX', 'MYX', 'NAX', 'NCX', 'NGX', 'NRX', 'OGX',
    'OKX', 'OMX', 'ONX', 'OPX', 'ORX', 'OTX', 'PBX', 'PDX', 'PHX', 'PIX', 'PLX', 'PNX', 'POX', 'PPX',
    'PQX', 'PSX', 'PTX', 'PUX', 'PVX', 'PWX', 'QRX', 'RAX', 'RDX', 'RIX', 'RNX', 'ROX', 'RSX', 'SAX',
    'SBX', 'SCX', 'SDX', 'SEX', 'SFX', 'SIX', 'SJX', 'SLX', 'SMX', 'SNX', 'SPX', 'SRX', 'STX', 'SXX',
    'SYX', 'TAX', 'TFX', 'TLX', 'TPX', 'TUX', 'TVX', 'TYX', 'UIX', 'VCX', 'XPX', 'YEX', 'YOX', 'YQX',
    'YUX', 'YVX', 'YYX', 'ZRX'
]

VALID_PASSENGER_CLASSES = ['Economy']

VALID_LOYALTY_LEVELS = ['NBK', 'global services', 'non-elite', 'premier 1k', 'premier gold', 'premier platinum', 'premier silver']

VALID_GENERATIONS = ['Boomer', 'Gen X', 'Gen Z', 'Millennial', 'NBK', 'Silent']

VALID_FLEET_TYPES = [
    'A319-100', 'A320-200', 'B737-700', 'B737-800', 'B737-900', 'B737-MAX8', 'B737-MAX9',
    'B757-200', 'B757-300', 'B767-300', 'B767-400', 'B777-200', 'B777-300', 'B787-10',
    'B787-8', 'B787-9', 'CRJ-550', 'CRJ-700', 'ERJ-170', 'ERJ-175'
]

AIRPORT_CODES = set(VALID_AIRPORT_CODES)
FLEET_TYPES = VALID_FLEET_TYPES + ['B777', 'B787', 'B737', 'B757', 'B767', 'A319', 'A320', 'A321', 'A330', 'A350', 'ERJ']
PASSENGER_CLASSES = [pclass.lower() for pclass in VALID_PASSENGER_CLASSES]  # Only 'economy'
LOYALTY_LEVELS = [level.lower() for level in VALID_LOYALTY_LEVELS]
GENERATIONS = [gen.lower() for gen in VALID_GENERATIONS] + ['baby boomer']

SATISFACTION_TERMS = ['satisfaction', 'rating', 'score', 'feedback', 'review', 'experience', 'rated']
DELAY_TERMS = ['delay', 'delayed', 'late', 'on-time', 'punctual', 'schedule']
ROUTE_TERMS = ['route', 'trip', 'journey', 'path', 'connection']

def extract_entities(text: str) -> Dict:
    """
    Extract airline-specific entities from text
    
    Args:
        text: User query
        
    Returns:
        Dictionary of extracted entities formatted for query templates
    """
    text_lower = text.lower()
    
    # Initialize entities based on database attributes
    entities = {
        # Database column entities
        'origin_station_code': None,
        'destination_station_code': None,
        'fleet_type_description': 'all',
        'generation': None,
        'loyalty_program_level': 'all',
        'passenger_class': 'all',
        
        # Query parameters for filtering
        'limit': 10,
        'min_satisfaction': None,
        'max_satisfaction': None,
        'max_delay': None,
        'min_legs': None,
        'min_miles': None,
        'max_miles': None,
        
        # Helper for single airport queries
        'airport_code': None
    }
    
    # Extract airport codes (3-letter uppercase patterns)
    airport_pattern = re.findall(r'\b([A-Z]{3})\b', text)
    valid_airports = [code for code in airport_pattern if code in AIRPORT_CODES]
    
    if valid_airports:
        # Detect "from X to Y" pattern for routes
        from_to_match = re.search(r'from\s+([A-Z]{3})\s+to\s+([A-Z]{3})', text, re.IGNORECASE)
        if from_to_match:
            origin = from_to_match.group(1).upper()
            dest = from_to_match.group(2).upper()
            if origin in AIRPORT_CODES:
                entities['origin_station_code'] = origin
            else:
                entities['origin_station_code'] = 'INVALID'
            if dest in AIRPORT_CODES:
                entities['destination_station_code'] = dest
            else:
                entities['destination_station_code'] = 'INVALID'
        elif len(valid_airports) >= 2:
            entities['origin_station_code'] = valid_airports[0]
            entities['destination_station_code'] = valid_airports[1]
        elif len(valid_airports) == 1:
            # Determine if origin or destination based on context
            if any(word in text_lower for word in ['from', 'departing', 'leaving']):
                entities['origin_station_code'] = valid_airports[0]
            elif any(word in text_lower for word in ['to', 'arriving', 'going']):
                entities['destination_station_code'] = valid_airports[0]
            else:
                entities['airport_code'] = valid_airports[0]
    
    # Extract flight numbers
    flight_match = re.findall(r'\bflight\s+(\d{2,4})\b', text_lower)
    if flight_match:
        entities['flight_number'] = int(flight_match[0])
    
    # Extract fleet types - check for both specific models and general types
    for fleet in FLEET_TYPES:
        if fleet.lower() in text_lower:
            entities['fleet_type_description'] = fleet
            break
    
    # Also check for general aircraft manufacturers
    if 'boeing' in text_lower and entities['fleet_type_description'] == 'all':
        # Look for Boeing model numbers
        boeing_match = re.search(r'boeing\s*(\d{3})', text_lower)
        if boeing_match:
            model = boeing_match.group(1)
            # Map common Boeing models
            boeing_models = {'737': 'B737', '747': 'B747', '757': 'B757', '767': 'B767', '777': 'B777', '787': 'B787'}
            if model in boeing_models:
                entities['fleet_type_description'] = boeing_models[model]
            else:
                entities['fleet_type_description'] = 'INVALID'
        else:
            entities['fleet_type_description'] = 'INVALID'
    
    elif 'airbus' in text_lower and entities['fleet_type_description'] == 'all':
        # Look for Airbus model numbers
        airbus_match = re.search(r'airbus\s*a?(\d{3})', text_lower)
        if airbus_match:
            model = airbus_match.group(1)
            # Validate if this Airbus model exists in our fleet types
            airbus_code = f'A{model}'
            if any(airbus_code in fleet for fleet in VALID_FLEET_TYPES):
                entities['fleet_type_description'] = airbus_code
            else:
                entities['fleet_type_description'] = 'INVALID'
        else:
            entities['fleet_type_description'] = 'INVALID'
    
    # Extract passenger class
    class_found = False
    for pclass in PASSENGER_CLASSES:
        if pclass in text_lower:
            entities['passenger_class'] = pclass
            class_found = True
            break
    
    # Check if passenger class was mentioned but not found in valid list
    if not class_found and any(word in text_lower for word in ['class', 'business', 'first', 'premium', 'economy', 'coach']):
        entities['passenger_class'] = 'INVALID'
    
    # Extract loyalty level
    loyalty_found = False
    for level in LOYALTY_LEVELS:
        if level in text_lower:
            entities['loyalty_program_level'] = level
            loyalty_found = True
            break
    
    # Check if loyalty level was mentioned but not found in valid list
    if not loyalty_found and any(word in text_lower for word in ['loyalty', 'elite', 'premier', 'member', 'status']):
        entities['loyalty_program_level'] = 'INVALID'
    
    # Extract generation
    gen_found = False
    for gen in GENERATIONS:
        if gen in text_lower:
            entities['generation'] = gen
            gen_found = True
            break
    
    # Check if generation was mentioned but not found in valid list
    if not gen_found and any(word in text_lower for word in ['generation', 'gen ', 'boomer', 'millennial', 'silent']):
        entities['generation'] = 'INVALID'
    
    # Set satisfaction thresholds based on query context
    satisfaction_mentioned = any(term in text_lower for term in SATISFACTION_TERMS)
    if satisfaction_mentioned:
        if any(word in text_lower for word in ['worst', 'poor', 'bad', 'low']):
            entities['max_satisfaction'] = 3  # Poor ratings
            entities['min_satisfaction'] = None
        elif any(word in text_lower for word in ['best', 'good', 'high', 'top']):
            entities['min_satisfaction'] = 3  # Good ratings
            entities['max_satisfaction'] = None
    # Also check for 'ratings' keyword
    elif 'rating' in text_lower:
        if any(word in text_lower for word in ['worst', 'poor', 'bad', 'low']):
            entities['max_satisfaction'] = 3
            entities['min_satisfaction'] = None
    
    # Set delay threshold if delay terms are mentioned
    if any(term in text_lower for term in DELAY_TERMS):
        # Extract specific delay numbers if mentioned
        delay_match = re.search(r'(\d+)\s*minutes?\s*delay', text_lower)
        if delay_match:
            entities['max_delay'] = int(delay_match.group(1))
        else:
            entities['max_delay'] = 30  # default threshold
    else:
        entities['max_delay'] = None
    
    # Set min_legs if multi-leg terms are mentioned  
    if any(term in text_lower for term in ['multi', 'leg', 'connecting', 'connection']):
        entities['min_legs'] = 2
    else:
        entities['min_legs'] = None
    
    # Extract miles-related queries
    if any(term in text_lower for term in ['miles', 'distance', 'long haul', 'short haul']):
        if any(word in text_lower for word in ['long', 'far', 'maximum']):
            entities['min_miles'] = 1000  # Long haul flights
        elif any(word in text_lower for word in ['short', 'close', 'minimum']):
            entities['max_miles'] = 1000  # Short haul flights
    
    # Extract numeric limits (top N, show N)
    limit_match = re.search(r'\b(top|show|first|limit)\s+(\d+)\b', text_lower)
    if limit_match:
        entities['limit'] = int(limit_match.group(2))
    
    return entities

def extract_entities_llm(query: str) -> Optional[Dict]:
    """
    Extract entities using LLM via OpenRouter API
    
    Args:
        query: User query string
        
    Returns:
        Dictionary of extracted entities or None if failed
    """
    if not OPENROUTER_API_KEY:
        print("OpenRouter API key not set.")
        return None
    
    system_prompt = f"""Extract airline entities from user queries. Return ONLY valid JSON with NO markdown.

IMPORTANT RULES:
1. Only extract SPECIFIC values mentioned in the query
2. Use null if the entity is NOT explicitly mentioned or cannot be determined
3. Valid generations: {', '.join(VALID_GENERATIONS)}
4. Valid loyalty levels: {', '.join(VALID_LOYALTY_LEVELS)}
5. Valid passenger class: Economy only
6. Valid airports: {', '.join(list(VALID_AIRPORT_CODES))}... (158 total)
7. Valid fleet types: {', '.join(VALID_FLEET_TYPES)}...

Examples:
- "Which generation travels most?" → generation: null (not specific)
- "How do Millennials travel?" → generation: "Millennial"
- "Show premier gold satisfaction" → loyalty_program_level: "premier gold"
- "Show loyalty satisfaction" → loyalty_program_level: null (not specific)

Return JSON:
{{
    "origin_station_code": "XXX" or null,
    "destination_station_code": "XXX" or null,
    "passenger_class": "Economy" or null,
    "loyalty_program_level": "specific level" or null,
    "generation": "specific generation" or null,
    "fleet_type_description": "specific type" or null,
    "limit": 10,
    "satisfaction_threshold": number or null,
    "delay_threshold": number or null
}}"""

    try:
        start = time.perf_counter()
        response = requests.post(
            url=OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract entities: '{query}'"}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            },
            timeout=30
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if response.status_code != 200:
            print("LLM API Error:", response.status_code, response.text)
            return None
        
        result = response.json()
        if 'choices' not in result or not result['choices']:
            print("LLM API returned no choices")
            return None
            
        llm_response = result['choices'][0]['message']['content'].strip()
        
        # Remove markdown code blocks if present
        if llm_response.startswith('```'):
            llm_response = re.sub(r'^```(?:json)?\s*\n?', '', llm_response)
            llm_response = re.sub(r'\n?```\s*$', '', llm_response)
            llm_response = llm_response.strip()
        
        entities = json.loads(llm_response)
        # Preserve original return type; attach metrics via a private attribute if needed elsewhere.
        # (Pipeline/UI will capture metrics via the new `extract_entities_llm_with_metrics` wrapper below.)
        return entities
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print("Using regex-based extraction as fallback.")
        return extract_entities(query)
    except Exception as e:
        print(f"Exception during LLM entity extraction: {e}")
        print("Using regex-based extraction as fallback.")
        return extract_entities(query)


def _normalize_usage_dict(usage: Optional[Dict]) -> Dict:
    if not usage or not isinstance(usage, dict):
        return {"prompt": None, "completion": None, "total": None}
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        try:
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        except Exception:
            total_tokens = None
    return {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens}


def extract_entities_llm_with_metrics(query: str) -> Dict:
    """LLM entity extraction returning {entities, metrics, error} without breaking legacy callers."""
    if not OPENROUTER_API_KEY:
        return {
            "entities": None,
            "metrics": {"response_time_ms": None, "tokens": {"prompt": None, "completion": None, "total": None}},
            "error": "OPENROUTER_API_KEY not set",
        }

    system_prompt = f"""Extract airline entities from user queries. Return ONLY valid JSON with NO markdown.

IMPORTANT RULES:
1. Only extract SPECIFIC values mentioned in the query
2. Use null if the entity is NOT explicitly mentioned or cannot be determined
3. Valid generations: {', '.join(VALID_GENERATIONS)}
4. Valid loyalty levels: {', '.join(VALID_LOYALTY_LEVELS)}
5. Valid passenger class: Economy only
6. Valid airports: {', '.join(list(VALID_AIRPORT_CODES))}... (158 total)
7. Valid fleet types: {', '.join(VALID_FLEET_TYPES)}...

Examples:
- "Which generation travels most?" → generation: null (not specific)
- "How do Millennials travel?" → generation: "Millennial"
- "Show premier gold satisfaction" → loyalty_program_level: "premier gold"
- "Show loyalty satisfaction" → loyalty_program_level: null (not specific)

Return JSON:
{{
    "origin_station_code": "XXX" or null,
    "destination_station_code": "XXX" or null,
    "passenger_class": "Economy" or null,
    "loyalty_program_level": "specific level" or null,
    "generation": "specific generation" or null,
    "fleet_type_description": "specific type" or null,
    "limit": 10,
    "satisfaction_threshold": number or null,
    "delay_threshold": number or null
}}"""

    start = time.perf_counter()
    try:
        response = requests.post(
            url=OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract entities: '{query}'"},
                ],
                "temperature": 0.1,
                "max_tokens": 500,
            },
            timeout=30,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if response.status_code != 200:
            return {
                "entities": None,
                "metrics": {
                    "response_time_ms": elapsed_ms,
                    "tokens": {"prompt": None, "completion": None, "total": None},
                },
                "error": f"LLM API Error: {response.status_code} {response.text}",
            }

        result = response.json()
        usage = _normalize_usage_dict(result.get("usage"))
        if "choices" not in result or not result["choices"]:
            return {
                "entities": None,
                "metrics": {"response_time_ms": elapsed_ms, "tokens": usage},
                "error": "LLM API returned no choices",
            }

        llm_response = (result["choices"][0]["message"]["content"] or "").strip()
        if llm_response.startswith("```"):
            llm_response = re.sub(r'^```(?:json)?\s*\n?', '', llm_response)
            llm_response = re.sub(r'\n?```\s*$', '', llm_response)
            llm_response = llm_response.strip()

        entities = json.loads(llm_response)
        return {
            "entities": entities,
            "metrics": {"response_time_ms": elapsed_ms, "tokens": usage},
            "error": None,
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "entities": None,
            "metrics": {"response_time_ms": elapsed_ms, "tokens": {"prompt": None, "completion": None, "total": None}},
            "error": str(e),
        }

if __name__ == "__main__":
    # Test queries - more realistic user inputs
    test_queries = [
        "Show me flights from LAX to JFK",
        "What are the top 5 busiest routes?",
        "Which flights have the worst ratings?", 
        "Find flights departing from EWX",
        "How satisfied are business class passengers?",
        "Compare premier gold vs non-elite satisfaction",
        "Show me delayed flights with more than 30 minutes delay",
        "What's the performance of Boeing 737 aircraft?",
        "Which generation travels the most on multi-leg journeys?",
        "Find the most punctual flights"
    ]
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        entities = extract_entities_llm(query)
        if entities:
            filtered_entities = {k: v for k, v in entities.items() 
                               if v is not None and v != [] and v != 'all'}
            print(f"Entities: {filtered_entities}")
        else:
            print("No entities extracted")