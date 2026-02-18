from neo4j import GraphDatabase
import pandas as pd
import argparse


def read_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == "":
                continue
            key, value = line.strip().split('=')
            config[key] = value
    return config 





def create_constraints(driver):
    with driver.session() as session:
        session.run("""
            CREATE CONSTRAINT passenger_record_locator IF NOT EXISTS
            FOR (p:Passenger) REQUIRE p.record_locator IS UNIQUE
        """)
        
        session.run("""
            CREATE CONSTRAINT journey_feedback_id IF NOT EXISTS
            FOR (j:Journey) REQUIRE j.feedback_ID IS UNIQUE
        """)
        
        session.run("""
            CREATE CONSTRAINT flight_number_fleet_type IF NOT EXISTS
            FOR (f:Flight) REQUIRE (f.flight_number, f.fleet_type_description) IS UNIQUE
        """)
        
        session.run("""
            CREATE CONSTRAINT airport_station_code IF NOT EXISTS
            FOR (a:Airport) REQUIRE a.station_code IS UNIQUE
        """)


# create the knowledge graph given the dataframe
def create_knowledge_graph(df, driver):
    """Create a knowledge graph in Neo4j from the given DataFrame
    """
    
    print("Total rows to process:", len(df))
    with driver.session() as session:
        for index, row in df.iterrows():
            session.run(
                """
                MERGE (p:Passenger {record_locator: $record_locator , loyalty_program_level: $loyalty_program_level, generation: $generation})
               
                
                MERGE (j:Journey {feedback_ID: $feedback_ID, food_satisfaction_score: $food_satisfaction_score, arrival_delay_minutes: $arrival_delay_minutes, actual_flown_miles: $actual_flown_miles, number_of_legs: $number_of_legs, passenger_class: $passenger_class})
                
                MERGE (f:Flight {flight_number: $flight_number, fleet_type_description: $fleet_type_description})
                
                MERGE (a1:Airport {station_code: $origin_station_code})
                MERGE (a2:Airport {station_code: $destination_station_code})
                
                MERGE (p)-[:TOOK]->(j)
                MERGE (j)-[:ON]->(f)
                MERGE (f)-[:DEPARTS_FROM]->(a1)
                MERGE (f)-[:ARRIVES_AT]->(a2)
                """,
                record_locator=row['record_locator'],
                loyalty_program_level=row['loyalty_program_level'],
                generation=row['generation'],
                feedback_ID=row['feedback_ID'],
                food_satisfaction_score=row['food_satisfaction_score'],
                arrival_delay_minutes=row['arrival_delay_minutes'],
                actual_flown_miles=row['actual_flown_miles'],
                number_of_legs=row['number_of_legs'],
                passenger_class=row['passenger_class'],
                flight_number=row['flight_number'],
                fleet_type_description=row['fleet_type_description'],
                origin_station_code=row['origin_station_code'],
                destination_station_code=row['destination_station_code']    
            )
            
            if (index + 1) % 1000 == 0:
                print(f"Processed {index + 1} rows...")
        print("\n")
       
    print("Knowledge graph created successfully.")
    
def test_query1(driver):
    with driver.session() as session:
        result = session.run("""
        MATCH (origin:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(destination:Airport)
        RETURN origin.station_code AS origin,
        destination.station_code AS destination,
        COUNT(f) AS flight_count
        ORDER BY flight_count DESC ,origin DESC 
        LIMIT 5
        """)
        print("Top 5 busiest flight routes:")
        print("Origin\tDestination\tFlight Count")
        print(result)
        for record in result:
            print(f"{record['origin']}\t{record['destination']}\t{record['flight_count']}")

def test_query2(driver):
    """Top 10 Flights with the Most Passenger Feedback"""
    with driver.session() as session:
        result = session.run("""
        MATCH (j:Journey)-[:ON]->(f:Flight)
        RETURN f.flight_number AS flight_id,
            COUNT(j) AS feedback_count
        ORDER BY feedback_count DESC
        LIMIT 10
        """)
        print("\nTop 10 Flights with the Most Passenger Feedback:")
        print(f"{'flight_id':<20} {'feedback_count':>15}")
        print("-" * 36)
        for record in result:
            print(f"{record['flight_id']:<20} {record['feedback_count']:>15}")

def test_query3(driver):
    """Average food satisfaction score for multi-leg journeys by generation"""
    with driver.session() as session:
        result = session.run("""
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)
        WHERE j.number_of_legs > 1
        RETURN p.generation AS generation,
               COUNT(j) AS multi_leg_count,
               AVG(j.food_satisfaction_score) AS avg_score
        ORDER BY multi_leg_count DESC
        """)
        print("\nAverage Food Satisfaction for Multi-Leg Journeys by Generation:")
        print(f"{'generation':<20} {'multi_leg_count':>15} {'avg_score':>12}")
        print("-" * 48)
        for record in result:
            print(f"{record['generation']:<20} {record['multi_leg_count']:>15} {record['avg_score']:>12.2f}")

def test_query4(driver):
    """Top 10 flights with shortest average arrival delays"""
    with driver.session() as session:
        result = session.run("""
        MATCH (j:Journey)-[:ON]->(f:Flight)
        RETURN f.flight_number AS flight_id,
               AVG(j.arrival_delay_minutes) AS avg_arrival_delay
        ORDER BY avg_arrival_delay ASC
        LIMIT 10
        """)
        print("\nTop 10 Flights with Shortest Average Arrival Delays:")
        print(f"{'flight_id':<20} {'avg_arrival_delay':>20}")
        print("-" * 41)
        for record in result:
            print(f"{record['flight_id']:<20} {record['avg_arrival_delay']:>20.2f}")

def test_query5(driver):
    """Average flown miles by loyalty program level"""
    with driver.session() as session:
        result = session.run("""
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)
        RETURN p.loyalty_program_level AS loyalty_level,
               AVG(j.actual_flown_miles) AS avg_actual_flown_miles
        ORDER BY avg_actual_flown_miles DESC
        """)
        print("\nAverage Flown Miles by Loyalty Program Level:")
        print(f"{'loyalty_level':<20} {'avg_actual_flown_miles':>25}")
        print("-" * 46)
        for record in result:
            print(f"{record['loyalty_level']:<20} {record['avg_actual_flown_miles']:>25.2f}")

def test_query6(driver):
    """Calculate overall satisfaction score and count passengers with score > 3"""
    with driver.session() as session:
        result = session.run("""
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)
        WITH j,
            j.food_satisfaction_score AS food_score, 
            5.0 - CASE 
                WHEN toFloat(round(abs(j.arrival_delay_minutes) / 20.0, 1)) > 5.0 THEN 5.0
                WHEN toFloat(round(abs(j.arrival_delay_minutes) / 20.0, 1)) < 0.0 THEN 0.0
                ELSE toFloat(round(abs(j.arrival_delay_minutes) / 20.0, 1))
            END AS delay_score,
            5.0 - CASE 
                WHEN toFloat(round(j.number_of_legs * 1.5, 1)) > 5.0 THEN 5.0
                WHEN toFloat(round(j.number_of_legs * 1.5, 1)) < 0.0 THEN 0.0
                ELSE toFloat(round(j.number_of_legs * 1.5, 1))
            END AS legs_score,
            5.0 - CASE 
                WHEN toFloat(round(j.actual_flown_miles / 3000.0, 1)) > 5.0 THEN 5.0
                WHEN toFloat(round(j.actual_flown_miles / 3000.0, 1)) < 0.0 THEN 0.0
                ELSE toFloat(round(j.actual_flown_miles / 3000.0, 1))
            END AS miles_score
    WITH j,
            round(0.5 * food_score + 
            0.35 * delay_score + 
            0.1 * legs_score + 
            0.05 * miles_score,1) AS overall_satisfaction_score
        WHERE overall_satisfaction_score > 3
        RETURN COUNT(j) AS passenger_count
        """)
        
        print("\nPassengers with Overall Satisfaction Score > 3:")
        print(f"{'passenger_count':>20}")
        print("-" * 20)
        for record in result:
            print(f"{record['passenger_count']:>20}")


def run_queries(driver):
    """Run all test queries"""
    
    print("\nRUNNING QUERIES")
    
    print("\nQuery 1 Results:")
    test_query1(driver)
    print("\n")
    print("Query 2 Results:")
    test_query2(driver)
    print("\n")
    print("Query 3 Results:")
    test_query3(driver)
    print("\n")
    print("Query 4 Results:")
    test_query4(driver)
    print("\n")
    print("Query 5 Results:")
    test_query5(driver)
    print("\n")
    print("Query 6 Results:")
    test_query6(driver)


def create_kg(driver, df):
    """Create the knowledge graph from scratch"""
    
    print("\nCREATING KNOWLEDGE GRAPH")
    
    # Clean up existing data if any
    print("\nCleaning up existing data...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("\nExisting data cleaned successfully.")
    
    # Create constraints for Primary Keys
    print("\nCreating constraints...")
    create_constraints(driver)
    print("\nConstraints created successfully.")
    
    # Create knowledge graph
    print("\nCreating knowledge graph...")
    create_knowledge_graph(df, driver)
    print("\nKnowledge graph creation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neo4j Knowledge Graph Manager')
    parser.add_argument('--create', action='store_true', help='Create the knowledge graph')
    parser.add_argument('--test', action='store_true', help='Run test queries')
    
    args = parser.parse_args()
    
    # If no arguments provided, run both
    if not args.create and not args.test:
        args.create = True
        args.test = True
    
    try:
        # Read configuration
        config = read_config("config.txt")
        URI = config.get("URI")  
        username = config.get("USERNAME", "neo4j")
        password = config.get("PASSWORD")  
        AUTH = (username, password)
        
        # Establish connection to Neo4j
        driver = GraphDatabase.driver(URI, auth=AUTH)
        
        # Verify connectivity
        driver.verify_connectivity()
        print("Connection established successfully.")

        # Load data from CSV
        df = pd.read_csv("Airline_surveys_sample.csv")
        
        if args.create:
            create_kg(driver, df)
        
        if args.test:
            run_queries(driver)
            
        # Close the driver connection
        driver.close()
            
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
    