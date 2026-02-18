QUERY_TEMPLATES = {
    
    'route': """
MATCH path = (origin:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(destination:Airport)
WHERE origin.station_code = $origin_station_code 
  AND destination.station_code = $destination_station_code
RETURN path
ORDER BY f.flight_number
LIMIT $limit
""",
    
    'popular_routes': """
MATCH (origin:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(destination:Airport)
WITH origin.station_code AS origin,
     destination.station_code AS destination,
     COUNT(DISTINCT f) AS flight_count
ORDER BY flight_count DESC
LIMIT $limit
MATCH path = (o:Airport {station_code: origin})<-[:DEPARTS_FROM]-(f2:Flight)-[:ARRIVES_AT]->(d:Airport {station_code: destination})
WITH origin, destination, flight_count, collect(path)[0] AS path
RETURN path, origin, destination, flight_count
ORDER BY flight_count DESC
""",

    'top_routes_satisfaction': """
MATCH (origin:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(destination:Airport),
      (j:Journey)-[:ON]->(f)
WHERE j.food_satisfaction_score >= $min_satisfaction
WITH origin.station_code AS origin_code, 
     destination.station_code AS dest_code,
     AVG(j.food_satisfaction_score) AS avg_satisfaction,
     COUNT(j) AS journey_count
ORDER BY avg_satisfaction DESC
LIMIT $limit
MATCH path = (origin:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(destination:Airport),
              (j:Journey)-[:ON]->(f)
WHERE origin.station_code = origin_code 
  AND destination.station_code = dest_code
  AND j.food_satisfaction_score >= $min_satisfaction
RETURN path, origin_code AS origin, dest_code AS destination, avg_satisfaction, journey_count
LIMIT $limit
""",

    'poorly_rated_flights': """
MATCH (j:Journey)-[:ON]->(f:Flight)
WHERE j.food_satisfaction_score <= $max_satisfaction
WITH f, AVG(j.food_satisfaction_score) AS avg_satisfaction, COUNT(j) AS journey_count
ORDER BY avg_satisfaction ASC
LIMIT $limit
MATCH path = (j:Journey)-[:ON]->(f)
WHERE j.food_satisfaction_score <= $max_satisfaction
RETURN path, avg_satisfaction, journey_count
LIMIT $limit
""",

    'flights_from_airport': """
MATCH path = (origin:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(destination:Airport)
WHERE origin.station_code = $origin_station_code
RETURN path
ORDER BY destination.station_code
LIMIT $limit
""",

    'flights_to_airport': """
MATCH path = (origin:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(destination:Airport)
WHERE destination.station_code = $destination_station_code
RETURN path
ORDER BY origin.station_code
LIMIT $limit
""",

    'flight_delay': """
MATCH (j:Journey)-[:ON]->(f:Flight)
WHERE j.arrival_delay_minutes <= $max_delay
WITH f,
     AVG(j.arrival_delay_minutes) AS avg_delay,
     COUNT(CASE WHEN j.arrival_delay_minutes <= 0 THEN 1 END) AS ontime_count,
     COUNT(j) AS total_count
ORDER BY avg_delay ASC
LIMIT $limit
MATCH path = (j:Journey)-[:ON]->(f)
WHERE j.arrival_delay_minutes <= $max_delay
RETURN path, avg_delay, ontime_count, total_count
LIMIT $limit
""",

    'satisfaction_by_class': """
MATCH (j:Journey)
WHERE j.passenger_class = $passenger_class OR $passenger_class = 'all'
WITH j.passenger_class AS class,
     AVG(j.food_satisfaction_score) AS avg_satisfaction,
     AVG(j.arrival_delay_minutes) AS avg_delay,
     COUNT(j) AS journey_count
ORDER BY avg_satisfaction DESC
LIMIT $limit
MATCH path = (j:Journey)
WHERE j.passenger_class = class OR $passenger_class = 'all'
RETURN path, class, avg_satisfaction, avg_delay, journey_count
LIMIT $limit
""",

    'satisfaction_by_loyalty': """
MATCH (p:Passenger)-[:TOOK]->(j:Journey)
WHERE p.loyalty_program_level = $loyalty_level OR $loyalty_level = 'all'
WITH p.loyalty_program_level AS loyalty_level,
     AVG(j.food_satisfaction_score) AS avg_satisfaction,
     AVG(j.arrival_delay_minutes) AS avg_delay,
     COUNT(j) AS journey_count
ORDER BY journey_count DESC
LIMIT $limit
MATCH path = (p:Passenger)-[:TOOK]->(j:Journey)
WHERE p.loyalty_program_level = loyalty_level OR $loyalty_level = 'all'
RETURN path, loyalty_level, avg_satisfaction, avg_delay, journey_count
LIMIT $limit
""",

    'fleet_performance': """
MATCH (j:Journey)-[:ON]->(f:Flight)
WHERE f.fleet_type_description CONTAINS $fleet_type OR $fleet_type = 'all'
WITH f.fleet_type_description AS fleet_type,
     AVG(j.food_satisfaction_score) AS avg_satisfaction,
     AVG(j.arrival_delay_minutes) AS avg_delay,
     COUNT(j) AS journey_count
ORDER BY avg_satisfaction DESC
LIMIT $limit
MATCH path = (j:Journey)-[:ON]->(f:Flight)
WHERE f.fleet_type_description = fleet_type OR $fleet_type = 'all'
RETURN path, fleet_type, avg_satisfaction, avg_delay, journey_count
LIMIT $limit
""",

    'multi_leg_insights': """
MATCH (p:Passenger)-[:TOOK]->(j:Journey)
WHERE j.number_of_legs >= $min_legs
WITH p.generation AS generation,
     AVG(j.number_of_legs) AS avg_legs,
     AVG(j.food_satisfaction_score) AS avg_satisfaction,
     AVG(j.arrival_delay_minutes) AS avg_delay,
     COUNT(j) AS journey_count
ORDER BY journey_count DESC
LIMIT $limit
MATCH path = (p:Passenger)-[:TOOK]->(j:Journey)
WHERE p.generation = generation AND j.number_of_legs >= $min_legs
RETURN path, generation, avg_legs, avg_satisfaction, avg_delay, journey_count
LIMIT $limit
"""
}


def get_query_template(intent: str) -> str:
    """
    Get Cypher query template for intent.
    
    Args:
        intent: Classified intent
        
    Returns:
        Cypher query template string, or None if intent is unknown
    """
    return QUERY_TEMPLATES.get(intent, None)


if __name__ == "__main__":
    # Quick test
    print("Available Query Templates:")
    print("=" * 50)
    for intent in QUERY_TEMPLATES.keys():
        print(f"\n{intent}:")
        print(QUERY_TEMPLATES[intent])
