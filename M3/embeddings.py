from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import os

# NOTE: Models are loaded lazily to keep imports lightweight (useful for Streamlit).
_MODEL1 = None
_MODEL2 = None


def get_model1() -> SentenceTransformer:
    global _MODEL1
    if _MODEL1 is None:
        _MODEL1 = SentenceTransformer("thenlper/gte-small")
    return _MODEL1


def get_model2() -> SentenceTransformer:
    global _MODEL2
    if _MODEL2 is None:
        token = os.getenv("HF_TOKEN", "")
        if not token:
            raise RuntimeError("HF_TOKEN is required to load google/embeddinggemma-300m")
        _MODEL2 = SentenceTransformer("google/embeddinggemma-300m", token=token)
    return _MODEL2

def _find_config_file(file_path: str = "config.txt"):
    candidates = [
        file_path,
        os.path.join(os.path.dirname(__file__), file_path),
        os.path.join(os.path.dirname(__file__), "config.txt"),
        os.path.join(os.path.dirname(__file__), "..", file_path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def read_config(file_path: str = "config.txt"):
    config = {}
    resolved = _find_config_file(file_path)
    if not resolved:
        return config
    with open(resolved, 'r') as file:
        for line in file:
            if line.strip() == "":
                continue
            key, value = line.strip().split('=', 1)
            config[key] = value
    return config 

def embed_nodes_in_neo4j(embedding_model):
    # creates embeddings for all nodes using all their properties as comma-separated text
    config = read_config("config.txt")
    URI = config.get("URI")  
    username = config.get("USERNAME", "neo4j")
    password = config.get("PASSWORD") 
    driver = GraphDatabase.driver(URI, auth=(username, password))

    with driver.session() as session:
        # get all nodes with ALL their properties
        result = session.run("""
            MATCH (n)
            RETURN id(n) as node_id, 
                   properties(n) as props,
                   labels(n)[0] as label
        """)
        
        nodes = []
        for record in result:
            # convert all properties to comma-separated text
            props = record['props']
            # filter out the embedding property if it already exists
            props_filtered = {k: v for k, v in props.items() if k not in ["id", "embedding1", "embedding2"]}
            # create text: "key1: value1, key2: value2, key3: value3"
            text_parts = [f"{key}: {value}" for key, value in props_filtered.items()]
            props_text  = ", ".join(text_parts)
            full_text = f"{record['label']}, {props_text}" if props_text else f"{record['label']}"
            nodes.append({
                'id': record['node_id'],
                'text': full_text,
                'label': record['label']
            })
        
        # generate embeddings
        print(f"Encoding {len(nodes)} nodes...")
        texts = [node['text'] for node in nodes]
        print("array to be encoded: ", texts)
        embeddings = embedding_model.encode(texts, convert_to_numpy=True)
        print(f"Sample node text: {texts[0][:200]}...")  # shows the first 200 characters
        
        # store embeddings back in Neo4j
        for i, node in enumerate(nodes):
            embedding_list = embeddings[i].tolist()
            if embedding_model is get_model1():
                session.run("""
                    MATCH (n)
                    WHERE id(n) = $node_id
                    SET n.embedding1 = $embedding1
                """, node_id=node['id'], embedding1=embedding_list)
            else:
                session.run("""
                    MATCH (n)
                    WHERE id(n) = $node_id
                    SET n.embedding2 = $embedding2
                """, node_id=node['id'], embedding2=embedding_list)
        if embedding_model is get_model1():
            print("Embeddings stored in Neo4j for model 1")
        else:
            print("Embeddings stored in Neo4j for model 2")

    driver.close()

def embed_nodes_and_relationships_in_neo4j(embedding_model):
    config = read_config("config.txt")
    URI = config.get("URI")  
    username = config.get("USERNAME", "neo4j")
    password = config.get("PASSWORD") 
    driver = GraphDatabase.driver(URI, auth=(username, password))

    with driver.session() as session:
        # get all nodes with their relationship context
        result = session.run("""
            MATCH (n)
            OPTIONAL MATCH (n)-[r]-(connected)
            WITH n, 
                 labels(n)[0] as label,
                 properties(n) as props,
                 collect({
                    rel_type: type(r),
                    connected_label: labels(connected)[0],
                    connected_props: properties(connected)
                 }) as connections
            RETURN id(n) as node_id, 
                   props,
                   label,
                   connections
        """)
        
        nodes = []
        for record in result:
            # filter out embeddings from main node properties
            props = record['props']
            props_filtered = {k: v for k, v in props.items() if k not in ["id", "embedding1", "embedding2", "relembed1", "relembed2"]}
            # build main node text
            text_parts = [f"{key}: {value}" for key, value in props_filtered.items()]
            props_text = ", ".join(text_parts)
            full_text = f"{record['label']}, {props_text}" if props_text else f"{record['label']}"
            
            # add relationship context
            connections = record['connections']
            if connections:  # checks if there are actual connections 
                context_parts = []
                for conn in connections:
                    if conn['rel_type']:  # skip null connections (rel_type is not null)
                        # get key properties from connected node
                        conn_props = conn['connected_props']
                        if conn_props:
                            # filter important properties (skip embeddings and id)
                            important_props = {k: v for k, v in conn_props.items() if k not in ["id", "embedding1", "embedding2", "relembed1", "relembed2"]}
                            # create a short description
                            if important_props:
                                prop_str = ", ".join([f"{k}: {v}" for k, v in important_props.items()])
                                context_parts.append(f"{conn['rel_type']} {conn['connected_label']} ({prop_str})")
                
                if context_parts:
                    full_text += " | Connected: " + "; ".join(context_parts)
            
            nodes.append({
                'id': record['node_id'],
                'text': full_text,
                'label': record['label']
            })
        
        # generate embeddings
        print(f"Encoding {len(nodes)} nodes...")
        texts = [node['text'] for node in nodes]
        print("Sample enriched text:", texts[0][:300], "...")
        embeddings = embedding_model.encode(texts, convert_to_numpy=True)
        
        # store embeddings back in Neo4j
        for i, node in enumerate(nodes):
            embedding_list = embeddings[i].tolist()
            if embedding_model is get_model1():
                session.run("""
                    MATCH (n)
                    WHERE id(n) = $node_id
                    SET n.relembed1 = $relembed1
                """, node_id=node['id'], relembed1=embedding_list)
            else:
                session.run("""
                    MATCH (n)
                    WHERE id(n) = $node_id
                    SET n.relembed2 = $relembed2
                """, node_id=node['id'], relembed2=embedding_list)
        
        print(f"Embeddings stored in Neo4j for {'model 1' if embedding_model is get_model1() else 'model 2'}")

    driver.close

def semantic_search(user_query, embedding_model, top_k, embedding_property: str = "embedding1"):
    query_embedding = embedding_model.encode([user_query], convert_to_numpy=True)[0]
    query_embedding_list = query_embedding.tolist()
    
    config = read_config("config.txt")
    URI = config.get("URI")  
    username = config.get("USERNAME", "neo4j")
    password = config.get("PASSWORD")  
    driver = GraphDatabase.driver(URI, auth=(username, password))
    
    property_name = embedding_property
    
    with driver.session() as session:
        # find matching nodes and their subgraphs
        result = session.run(f"""
            MATCH (n)
            WHERE n.{property_name} IS NOT NULL
            WITH n, 
                gds.similarity.cosine(n.{property_name}, $query_embedding) AS similarity
            WHERE similarity > 0
            ORDER BY similarity DESC
            LIMIT $top_k
            
            OPTIONAL MATCH (n)-[r]-(connected)
            WITH n, similarity, 
                 elementId(n) as matched_id,
                 labels(n)[0] as matched_type,
                 properties(n) as matched_props,
                 collect({{
                    relationship_type: type(r),
                    from_node_id: elementId(startNode(r)), 
                    to_node_id: elementId(endNode(r)), 
                    connected_node_id: elementId(connected),
                    connected_node_type: labels(connected)[0],
                    connected_node_properties: properties(connected)
                 }}) as connections
            RETURN matched_id,
                   matched_type,
                   similarity,
                   matched_props,
                   connections
        """, query_embedding=query_embedding_list, top_k=top_k)
        
        matched_nodes = {}  # nodes (high similarity)
        related_nodes = {}  # connected nodes 
        relationships = {}
        
        for record in result:
            matched_props = {k: v for k, v in record['matched_props'].items() if k not in ['embedding1', 'embedding2', "relembed1", "relembed2"]}
            matched_id = record['matched_id']
            # store matched node (with similarity score)
            if matched_id not in matched_nodes:
                matched_nodes[matched_id] = {
                    'id': matched_id,
                    'type': record['matched_type'],
                    'properties': matched_props,
                    'similarity_score': record['similarity']
                }
            
            # process connections
            for conn in record['connections']:
                if conn['connected_node_type'] is not None:
                    conn_props = {k: v for k, v in conn['connected_node_properties'].items() if k not in ['embedding1', 'embedding2', "relembed1", "relembed2"]}
                    conn_id = conn['connected_node_id']          
                    # store connected node (without similarity if not already stored)
                    if conn_id not in matched_nodes and conn_id not in related_nodes:
                        related_nodes[conn_id] = {
                            'id': conn_id,
                            'type': conn['connected_node_type'],
                            'properties': conn_props
                        }

                    rel_key = (conn['from_node_id'], conn['to_node_id'], conn['relationship_type'])
            
                    # store relationship (only if not already stored)
                    if rel_key not in relationships:
                        relationships[rel_key] = {
                            'relationship_type': conn['relationship_type'],
                            'from_node_id': conn['from_node_id'], 
                            'to_node_id': conn['to_node_id']
                        }                    
        
        return {
            'nodes': list(matched_nodes.values()),
            'rel_nodes': list(related_nodes.values()),
            'relationships': list(relationships.values())
        }

def semantic_search2(user_query, embedding_model, top_k, embedding_property: str = "relembed1"):
    query_embedding = embedding_model.encode([user_query], convert_to_numpy=True)[0]
    query_embedding_list = query_embedding.tolist()
    
    config = read_config("config.txt")
    URI = config.get("URI")  
    username = config.get("USERNAME", "neo4j")
    password = config.get("PASSWORD")  
    driver = GraphDatabase.driver(URI, auth=(username, password))
    
    property_name = embedding_property
    
    with driver.session() as session:
        # find matching nodes and their subgraphs
        result = session.run(f"""
            MATCH (n)
            WHERE n.{property_name} IS NOT NULL
            WITH n, 
                gds.similarity.cosine(n.{property_name}, $query_embedding) AS similarity
            WHERE similarity > 0
            ORDER BY similarity DESC
            LIMIT $top_k
            
            OPTIONAL MATCH (n)-[r]-(connected)
            WITH n, similarity, 
                 elementId(n) as matched_id,
                 labels(n)[0] as matched_type,
                 properties(n) as matched_props,
                 collect({{
                    relationship_type: type(r),
                    from_node_id: elementId(startNode(r)),
                    to_node_id: elementId(endNode(r)), 
                    connected_node_id: elementId(connected),
                    connected_node_type: labels(connected)[0],
                    connected_node_properties: properties(connected)
                 }}) as connections
            RETURN matched_id,
                   matched_type,
                   similarity,
                   matched_props,
                   connections
        """, query_embedding=query_embedding_list, top_k=top_k)
        
        matched_nodes = {}  # nodes (high similarity)
        related_nodes = {}  # connected nodes 
        relationships = {}
        
        for record in result:
            matched_props = {k: v for k, v in record['matched_props'].items() if k not in ['embedding1', 'embedding2', "relembed1", "relembed2"]}
            matched_id = record['matched_id']
            # store matched node (with similarity score)
            if matched_id not in matched_nodes:
                matched_nodes[matched_id] = {
                    'id': matched_id,
                    'type': record['matched_type'],
                    'properties': matched_props,
                    'similarity_score': record['similarity']
                }
            
            # process connections
            for conn in record['connections']:
                if conn['connected_node_type'] is not None:
                    conn_props = {k: v for k, v in conn['connected_node_properties'].items() if k not in ['embedding1', 'embedding2', "relembed1", "relembed2"]}
                    conn_id = conn['connected_node_id']          
                    # store connected node (without similarity if not already stored)
                    if conn_id not in matched_nodes and conn_id not in related_nodes:
                        related_nodes[conn_id] = {
                            'id': conn_id,
                            'type': conn['connected_node_type'],
                            'properties': conn_props
                        }
                    
                    rel_key = (conn['from_node_id'], conn['to_node_id'], conn['relationship_type'])
            
                    # store relationship (only if not already stored)
                    if rel_key not in relationships:
                        relationships[rel_key] = {
                            'relationship_type': conn['relationship_type'],
                            'from_node_id': conn['from_node_id'], 
                            'to_node_id': conn['to_node_id']
                        }    
        
        return {
            'nodes': list(matched_nodes.values()),
            'rel_nodes': list(related_nodes.values()),
            'relationships': list(relationships.values())
        }

if __name__ == "__main__":

    print("hi")

    # print("Embedding nodes using Model 1...")
    # embed_nodes_in_neo4j(get_model1())

    # print("Embedding nodes using Model 2...")
    # embed_nodes_in_neo4j(get_model2())

    # print("Embedding nodes and relationships using Model 1...")
    # embed_nodes_and_relationships_in_neo4j(get_model1())

    # print("Embedding nodes and relationships using Model 2...")
    # embed_nodes_and_relationships_in_neo4j(get_model2())

    query1 = "Which flight was delayed the most?"
    print(f"Searching for: {query1}")

    print(f"Searching for: {query1} with model 1 only embedding the node")
    results1 = semantic_search(query1, get_model1(), 3, embedding_property="embedding1")
    print("Matched nodes:")
    for node in results1["nodes"]:
        print(node)
    print("==================================================")
    print("Related nodes:")
    for relNode in results1["rel_nodes"]:
        print(relNode)
    print("==================================================")
    print("Relationships:")
    for rel in results1["relationships"]:
        print(rel)
    print("==================================================")

    print(f"Searching for: {query1} with model 2 only embedding the node")
    # results2 = semantic_search(query1, get_model2(), 3, embedding_property="embedding2")
    print("Matched nodes:")
    for node in results2["nodes"]:
        print(node)
    print("==================================================")
    print("Related nodes:")
    for relNode in results2["rel_nodes"]:
        print(relNode)
    print("==================================================")
    print("Relationships:")
    for rel in results2["relationships"]:
        print(rel)
    print("==================================================")

    print(f"Searching for: {query1} with model 1 embedding the each node with its relationships")
    results3 = semantic_search2(query1, get_model1(), 3, embedding_property="relembed1")
    print("Matched nodes:")
    for node in results3["nodes"]:
        print(node)
    print("==================================================")
    print("Related nodes:")
    for relNode in results3["rel_nodes"]:
        print(relNode)
    print("==================================================")
    print("Relationships:")
    for rel in results3["relationships"]:
        print(rel)
    print("==================================================")

    print(f"Searching for: {query1} with model 1 embedding the each node with its relationships")
    # results4 = semantic_search2(query1, get_model2(), 3, embedding_property="relembed2")
    print("Matched nodes:")
    for node in results4["nodes"]:
        print(node)
    print("==================================================")
    print("Related nodes:")
    for relNode in results4["rel_nodes"]:
        print(relNode)
    print("==================================================")
    print("Relationships:")
    for rel in results4["relationships"]:
        print(rel)
    print("==================================================")




    query2 = "Which passengers were most satisfied with the food"
    print(f"Searching for: {query2}")

    print(f"Searching for: {query2} with model 1 only embedding the node")
    results5 = semantic_search(query2, get_model1(), 3, embedding_property="embedding1")
    print("Matched nodes:")
    for node in results5["nodes"]:
        print(node)
    print("==================================================")
    print("Related nodes:")
    for relNode in results5["rel_nodes"]:
        print(relNode)
    print("==================================================")
    print("Relationships:")
    for rel in results5["relationships"]:
        print(rel)
    print("==================================================")

    print(f"Searching for: {query2} with model 2 only embedding the node")
    # results6 = semantic_search(query2, get_model2(), 3, embedding_property="embedding2")
    print("Matched nodes:")
    for node in results6["nodes"]:
        print(node)
    print("==================================================")
    print("Related nodes:")
    for relNode in results6["rel_nodes"]:
        print(relNode)
    print("==================================================")
    print("Relationships:")
    for rel in results6["relationships"]:
        print(rel)
    print("==================================================")

    print(f"Searching for: {query2} with model 1 embedding the each node with its relationships")
    results7 = semantic_search2(query2, get_model1(), 3, embedding_property="relembed1")
    print("Matched nodes:")
    for node in results7["nodes"]:
        print(node)
    print("==================================================")
    print("Related nodes:")
    for relNode in results7["rel_nodes"]:
        print(relNode)
    print("==================================================")
    print("Relationships:")
    for rel in results7["relationships"]:
        print(rel)
    print("==================================================")

    print(f"Searching for: {query2} with model 1 embedding the each node with its relationships")
    # results8 = semantic_search2(query2, get_model2(), 3, embedding_property="relembed2")
    print("Matched nodes:")
    for node in results8["nodes"]:
        print(node)
    print("==================================================")
    print("Related nodes:")
    for relNode in results8["rel_nodes"]:
        print(relNode)
    print("==================================================")
    print("Relationships:")
    for rel in results8["relationships"]:
        print(rel)
    print("==================================================")




    query3 = "Which airports do non-elite members go to"
    print(f"Searching for: {query3}")

    print(f"Searching for: {query3} with model 1 only embedding the node")
    results9 = semantic_search(query3, get_model1(), 3, embedding_property="embedding1")
    print("Matched nodes:")
    for node in results9["nodes"]:
        print(node)
    print("==================================================")
    print("Related nodes:")
    for relNode in results9["rel_nodes"]:
        print(relNode)
    print("==================================================")
    print("Relationships:")
    for rel in results9["relationships"]:
        print(rel)
    print("==================================================")

    print(f"Searching for: {query3} with model 2 only embedding the node")
    # results10 = semantic_search(query3, get_model2(), 3, embedding_property="embedding2")
    print("Matched nodes:")
    for node in results10["nodes"]:
        print(node)
    print("==================================================")
    print("Related nodes:")
    for relNode in results10["rel_nodes"]:
        print(relNode)
    print("==================================================")
    print("Relationships:")
    for rel in results10["relationships"]:
        print(rel)
    print("==================================================")

    print(f"Searching for: {query3} with model 1 embedding the each node with its relationships")
    results11 = semantic_search2(query3, get_model1(), 3, embedding_property="relembed1")
    print("Matched nodes:")
    for node in results11["nodes"]:
        print(node)
    print("==================================================")
    print("Related nodes:")
    for relNode in results11["rel_nodes"]:
        print(relNode)
    print("==================================================")
    print("Relationships:")
    for rel in results11["relationships"]:
        print(rel)
    print("==================================================")

    print(f"Searching for: {query3} with model 1 embedding the each node with its relationships")
    # results12 = semantic_search2(query3, get_model2(), 3, embedding_property="relembed2")
    print("Matched nodes:")
    for node in results12["nodes"]:
        print(node)
    print("==================================================")
    print("Related nodes:")
    for relNode in results12["rel_nodes"]:
        print(relNode)
    print("==================================================")
    print("Relationships:")
    for rel in results12["relationships"]:
        print(rel)
    print("==================================================")