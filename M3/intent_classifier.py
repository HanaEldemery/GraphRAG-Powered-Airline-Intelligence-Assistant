import re

INTENT_KEYWORDS = {
    'popular_routes': ['busiest route', 'popular route', 'most traveled', 'busiest', 'most popular', 'top routes', 'most frequent'],
    'top_routes_satisfaction': ['best route', 'top rated route', 'highest satisfaction route', 'best performing route', 'route satisfaction', 'routes satisfaction', 'highest satisfaction'],
    'poorly_rated_flights': ['poorly rated', 'worst flight', 'bad flight', 'low rating', 'worst rated', 'lowest satisfaction', 'worst performing', 'worst rating','worst ratings', 'bad rating'],
    'flights_from_airport': ['flights from', 'departing from', 'leaving from', 'depart from', 'flights leaving'],
    'flights_to_airport': ['flights to', 'arriving at', 'going to', 'arrive at', 'flights arriving'],
    'flight_delay': ['minimal delay', 'least delay', 'shortest delay', 'on time', 'on-time', 'punctual', 'delay','delays'],
    'satisfaction_by_class': ['class satisfaction', 'economy satisfaction', 'business class', 'passenger class', 'class comparison', 'by class', 'travel class', 'satisfaction class'],
    'satisfaction_by_loyalty': ['loyalty satisfaction', 'loyalty level','loyalty' 'loyalty program', 'elite satisfaction', 'premier', 'gold members', 'by loyalty'],
    'fleet_performance': ['fleet performance', 'aircraft performance', 'fleet type', 'aircraft type', 'boeing', 'airbus', 'plane type'],
    'multi_leg_insights': ['multi-leg', 'multi leg', 'connecting flight', 'layover', 'connection', 'multiple legs'],
    'route': ['route from', 'route to', r'\bfrom\b', r'\bto\b', 'between', 'flight from', 'flight to', 'flights between'],
}

def classify_intent(user_input: str) -> str:
    """
    Classify user intent using keyword matching with word boundaries.
    
    Args:
        user_input: User query string
        
    Returns:
        Intent string that maps to a query template, or 'unknown' if no match
    """
    user_input_lower = user_input.lower()
    
    # Score each intent based on keyword matches
    intent_scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            # Use word boundary matching for single words, phrase matching for multi-word
            if keyword.startswith(r'\b'):
                # Regex pattern with word boundaries
                if re.search(keyword, user_input_lower):
                    score += 1
            else:
                # Multi-word phrase - use word boundary for whole phrase
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, user_input_lower):
                    score += 1
        
        if score > 0:
            intent_scores[intent] = score
    
    # Return intent with highest score if any match found
    if intent_scores:
        return max(intent_scores.items(), key=lambda x: x[1])[0]
    
    # Return unknown if no intent matched
    return 'unknown'

if __name__ == "__main__":
    # Quick test
    test_queries = [
        "Show me the busiest routes",
        "What are the most delayed flights?",
        "Show flights from LAX",
        "Which flights have the worst ratings?",
        "How does economy class satisfaction compare?",
        "Show fleet performance by aircraft type",
        "What about multi-leg journey insights?",
        "Random unrelated query about weather",
        "what is the whether today",
    ]
    
    for query in test_queries:
        intent = classify_intent(query)
        print(f"Query: {query}")
        print(f"  Intent: {intent}")
        print()
