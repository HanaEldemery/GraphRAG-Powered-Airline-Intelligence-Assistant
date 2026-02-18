from intent_classifier import classify_intent
from query_templates import get_query_template, QUERY_TEMPLATES
from neo4j import GraphDatabase


NEO4J_URI = ""
NEO4J_USER = ""
NEO4J_PASSWORD = ""


# Entity placeholders for each intent
TEST_ENTITIES = {
    'popular_routes': {'limit': 5},
    'top_routes_satisfaction': {'min_satisfaction': 3, 'limit': 5},
    'poorly_rated_flights': {'max_satisfaction': 3, 'limit': 5},
    'flights_from_airport': {'airport_code': 'LAX', 'limit': 5},
    'flights_to_airport': {'airport_code': 'JFK', 'limit': 5},
    'flight_delay': {'max_delay': 30, 'limit': 5},
    'satisfaction_by_class': {'passenger_class': 'all', 'limit': 10},
    'satisfaction_by_loyalty': {'loyalty_level': 'all', 'limit': 10},
    'fleet_performance': {'fleet_type': 'all', 'limit': 5},
    'multi_leg_insights': {'min_legs': 2, 'limit': 5}
}

# Test queries for each intent
TEST_QUERIES = [
    ("Show me the busiest routes", "popular_routes"),
    ("Which routes have the highest satisfaction?", "top_routes_satisfaction"),
    ("Which flights have the worst ratings?", "poorly_rated_flights"),
    ("Show flights departing from LAX", "flights_from_airport"),
    ("Find flights arriving at JFK", "flights_to_airport"),
    ("Which flights have the shortest delays?", "flight_delay"),
    ("Compare satisfaction by travel class", "satisfaction_by_class"),
    ("How does loyalty level affect satisfaction?", "satisfaction_by_loyalty"),
    ("What is the B777 fleet performance?", "fleet_performance"),
    ("Show multi-leg journey patterns", "multi_leg_insights"),
    ("What's the weather like today?", "unknown"),  # Test unknown intent
]


def test_intent_classification():
    """Test intent classification accuracy."""
    print("=" * 60)
    print("TESTING INTENT CLASSIFICATION")
    print("=" * 60)
    
    correct = 0
    for query, expected in TEST_QUERIES:
        intent = classify_intent(query)
        status = "✓" if intent == expected else "✗"
        if intent == expected:
            correct += 1
        print(f"{status} '{query}'\n   -> {intent} (expected: {expected})")
    
    print(f"\nAccuracy: {correct}/{len(TEST_QUERIES)} ({100*correct/len(TEST_QUERIES):.0f}%)")


def test_full_pipeline():
    """Test complete pipeline: user query -> intent -> database result."""
    print("\n" + "=" * 60)
    print("TESTING FULL PIPELINE WITH DATABASE")
    print("=" * 60)
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return
    
    test_cases = [
        "Show me the busiest routes",
        "Flights departing from LAX",
        "How do loyalty levels compare?",
        "What is the weather?",  # Should get unknown
    ]
    
    with driver.session() as session:
        for user_query in test_cases:
            print(f"\nUser Query: '{user_query}'")
            
            # Step 1: Classify intent
            intent = classify_intent(user_query)
            print(f"Classified Intent: {intent}")
            
            # Step 2: Get query template
            template = get_query_template(intent)
            
            if template is None:
                print("No query template available for this intent.")
            else:
                entities = TEST_ENTITIES.get(intent, {})
                print(f"Entities: {entities}")
                
                try:
                    result = session.run(template, **entities)
                    records = list(result)
                    print(f"Results: {len(records)} rows")
                    for record in records[:3]:
                        print(f"  -> {dict(record)}")
                except Exception as e:
                    print(f"Error: {e}")
    
    driver.close()


if __name__ == "__main__":
    test_intent_classification()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    test_full_pipeline()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
