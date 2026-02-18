try:
    from .intent_classifier import classify_intent
    from .entity_extractor import extract_entities, extract_entities_llm_with_metrics
    from .query_templates import get_query_template
except ImportError:  # allows running as a script from inside the M3 folder
    from intent_classifier import classify_intent
    from entity_extractor import extract_entities, extract_entities_llm_with_metrics
    from query_templates import get_query_template
from neo4j import GraphDatabase
from typing import Any, Dict, Iterable, List, Optional, Tuple
import os
import json

def _find_config_file(file_path: str = "config.txt") -> Optional[str]:
    """Locate config file across common run contexts (repo root vs M3 folder)."""
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


def read_config(file_path: str = "config.txt") -> Dict[str, str]:
    """Read database configuration from config file (if present)."""
    resolved = _find_config_file(file_path)
    if not resolved:
        return {}

    config: Dict[str, str] = {}
    with open(resolved, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                key, value = line.strip().split("=", 1)
                config[key] = value
    return config

def _get_neo4j_credentials() -> Tuple[str, str, str]:
    """Resolve Neo4j credentials from env vars (preferred) or config file."""
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if uri and username and password:
        return uri, username, password

    config = read_config()
    if not uri:
        uri = config.get("URI")
    if not username:
        username = config.get("USERNAME")
    if not password:
        password = config.get("PASSWORD")

    if not uri or not username or not password:
        raise RuntimeError(
            "Neo4j credentials not found. Set NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD "
            "or provide a config.txt (e.g., M3/config.txt)."
        )

    return uri, username, password


def connect_to_neo4j() -> GraphDatabase.driver:
    """Connect to Neo4j database using env vars or config file."""
    uri, username, password = _get_neo4j_credentials()
    return GraphDatabase.driver(uri, auth=(username, password))


def _is_path_like(value: Any) -> bool:
    return hasattr(value, "nodes") and hasattr(value, "relationships")


def _extract_paths_from_record(record: Any) -> List[Any]:
    """Extract all Neo4j paths from a Record, regardless of return variable name."""
    paths: List[Any] = []
    try:
        keys = list(record.keys())
    except Exception:
        keys = []

    for key in keys:
        try:
            value = record.get(key)
        except Exception:
            continue
        if _is_path_like(value):
            paths.append(value)
    return paths


def _path_to_subgraph(path: Any) -> Dict[str, Any]:
    keys_to_hide = {"embedding1", "embedding2", "relembed1", "relembed2"}
    return {
        "nodes": [
            {
                "id": node.element_id,
                "labels": list(node.labels),
                "properties": {k: v for k, v in dict(node).items() if k not in keys_to_hide},
            }
            for node in path.nodes
        ],
        "relationships": [
            {
                "type": rel.type,
                "start_node": rel.start_node.element_id,
                "end_node": rel.end_node.element_id,
            }
            for rel in path.relationships
        ],
    }


def _jsonify_scalar(value: Any) -> Any:
    """Best-effort conversion of Neo4j return scalars into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if _is_path_like(value):
        return None
    if isinstance(value, list):
        return [_jsonify_scalar(v) for v in value[:50]]
    if isinstance(value, dict):
        return {str(k): _jsonify_scalar(v) for k, v in list(value.items())[:50]}

    # Neo4j temporal types / other objects often support isoformat().
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:
            pass

    # Fallback: keep it readable.
    try:
        return str(value)
    except Exception:
        return None


def _merge_subgraphs(subgraphs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    rel_keys: set[Tuple[str, str, str]] = set()
    relationships: List[Dict[str, Any]] = []

    for sg in subgraphs:
        for node in sg.get("nodes", []):
            nodes_by_id.setdefault(node["id"], node)

        for rel in sg.get("relationships", []):
            key = (rel["start_node"], rel["end_node"], rel["type"])
            if key in rel_keys:
                continue
            rel_keys.add(key)
            relationships.append(rel)

    return {"nodes": list(nodes_by_id.values()), "relationships": relationships}


def _merge_graphs(graphs: Iterable[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    rel_keys: set[Tuple[str, str, str]] = set()
    relationships: List[Dict[str, Any]] = []

    for graph in graphs:
        if not graph:
            continue
        for node in graph.get("nodes", []) or []:
            node_id = node.get("id")
            if not node_id:
                continue
            nodes_by_id.setdefault(node_id, node)

        for rel in graph.get("relationships", []) or []:
            start_node = rel.get("start_node")
            end_node = rel.get("end_node")
            rel_type = rel.get("type")
            if not start_node or not end_node or not rel_type:
                continue
            key = (start_node, end_node, rel_type)
            if key in rel_keys:
                continue
            rel_keys.add(key)
            relationships.append({"type": rel_type, "start_node": start_node, "end_node": end_node})

    return {"nodes": list(nodes_by_id.values()), "relationships": relationships}


def _embeddings_trace_to_graph(embedding_trace: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert embeddings module output (hits/related_nodes/relationships) into graph format."""
    if not embedding_trace or embedding_trace.get("error"):
        return None

    nodes_by_id: Dict[str, Dict[str, Any]] = {}

    for hit in embedding_trace.get("hits", []) or []:
        node_id = hit.get("id")
        if not node_id:
            continue
        props = dict(hit.get("properties") or {})
        if "similarity_score" in hit:
            props["similarity_score"] = hit.get("similarity_score")
        nodes_by_id.setdefault(
            node_id,
            {
                "id": node_id,
                "labels": [hit.get("type") or "Node"],
                "properties": props,
            },
        )

    for rel_node in embedding_trace.get("related_nodes", []) or []:
        node_id = rel_node.get("id")
        if not node_id:
            continue
        nodes_by_id.setdefault(
            node_id,
            {
                "id": node_id,
                "labels": [rel_node.get("type") or "Node"],
                "properties": dict(rel_node.get("properties") or {}),
            },
        )

    relationships: List[Dict[str, Any]] = []
    rel_keys: set[Tuple[str, str, str]] = set()
    for rel in embedding_trace.get("relationships", []) or []:
        rel_type = rel.get("relationship_type")
        start_node = rel.get("from_node_id")
        end_node = rel.get("to_node_id")
        if not rel_type or not start_node or not end_node:
            continue
        key = (start_node, end_node, rel_type)
        if key in rel_keys:
            continue
        rel_keys.add(key)
        relationships.append({"type": rel_type, "start_node": start_node, "end_node": end_node})

    if not nodes_by_id and not relationships:
        return None
    return {"nodes": list(nodes_by_id.values()), "relationships": relationships}

def process_query(query: str, use_llm: bool = True, execute: bool = False, verbose: bool = True) -> Dict:
    """
    Process user query through the complete pipeline
    
    Args:
        query: User's natural language query
        use_llm: Whether to attempt LLM extraction first (falls back to regex)
        execute: Whether to execute the query against Neo4j database
    
    Returns:
        dict containing intent, entities, cypher_query, extraction_method, and results
    """
    if verbose:
        print("=" * 80)
        print(f"USER QUERY: {query}")
        print("=" * 80)
    
    # Step 1: Classify intent
    intent = classify_intent(query)
    if verbose:
        print(f"\nINTENT: {intent}")
    
    # Step 2: Extract entities (LLM with regex fallback)
    entities = None
    extraction_method = 'regex'
    entity_extraction_metrics: Optional[Dict[str, Any]] = None
    
    if use_llm:
        llm_entity = extract_entities_llm_with_metrics(query)
        entity_extraction_metrics = llm_entity.get("metrics") if isinstance(llm_entity, dict) else None
        entities = llm_entity.get("entities") if isinstance(llm_entity, dict) else None
        if entities:
            extraction_method = 'llm'
    
    if not entities:
        entities = extract_entities(query)
        extraction_method = 'regex'
    
    if verbose:
        print(f"EXTRACTION METHOD: {extraction_method}")
    
    # Filter and show extracted entities
    relevant_entities = {k: v for k, v in entities.items() 
                        if v is not None and v != 'all' and v != 10 and v != []}
    if verbose:
        print(f"EXTRACTED ENTITIES: {relevant_entities if relevant_entities else 'None'}")
    
    # Step 3: Override intent if both origin and destination are present (route query)
    if (entities.get('origin_station_code') and 
        entities.get('destination_station_code') and
        entities.get('origin_station_code') != 'INVALID' and
        entities.get('destination_station_code') != 'INVALID'):
        intent = 'route'
        if verbose:
            print(f"INTENT ADJUSTED TO: {intent} (both origin and destination detected)")
    
    # Step 4: Get query template
    template = get_query_template(intent)
    if verbose:
        print("\nQUERY TEMPLATE:")
        print(template.strip() if template else "No template found")
    
    # Step 5: Fill template with entities
    cypher_query = fill_template(template, entities)
    if verbose:
        print("\nFILLED QUERY:")
        print(cypher_query.strip() if cypher_query else "No query generated")
    
    # Step 6: Execute query if requested
    results = None
    cypher_rows: Optional[List[Dict[str, Any]]] = None
    if execute and cypher_query:
        if verbose:
            print(f"\n{'=' * 80}")
            print("EXECUTING QUERY...")
            print("=" * 80)
        try:
            driver = connect_to_neo4j()
            with driver.session() as session:
                result = session.run(cypher_query)
                results = []
                cypher_rows = []

                for record in result:
                    # Capture any returned scalar values (e.g., COUNT/AVG metrics)
                    row: Dict[str, Any] = {}
                    try:
                        keys = list(record.keys())
                    except Exception:
                        keys = []
                    for key in keys:
                        try:
                            value = record.get(key)
                        except Exception:
                            continue
                        if _is_path_like(value):
                            continue
                        json_value = _jsonify_scalar(value)
                        if json_value is None:
                            continue
                        row[str(key)] = json_value
                    if row:
                        cypher_rows.append(row)

                    paths = _extract_paths_from_record(record)
                    for path in paths:
                        sub_graph = _path_to_subgraph(path)
                        if verbose:
                            print("\n")
                            print(sub_graph)
                        results.append(sub_graph)
                
            driver.close()
        except Exception as e:
            if verbose:
                print(f"ERROR EXECUTING QUERY: {e}")
            results = {'error': str(e)}
    
    return {
        'query': query,
        'intent': intent,
        'entities': entities,
        'cypher_query': cypher_query,
        'extraction_method': extraction_method,
        'results': results,
        'merged_graph': _merge_subgraphs(results) if isinstance(results, list) else None,
        'cypher_rows': cypher_rows,
        'metrics': {
            'entity_extraction': entity_extraction_metrics
        }
    }


def _build_context_text(
    merged_graph: Optional[Dict[str, Any]],
    *,
    cypher_query: Optional[str] = None,
    cypher_rows: Optional[List[Dict[str, Any]]] = None,
    max_chars: int = 8000,
) -> str:
    if not merged_graph and not cypher_query:
        return ""

    noisy_keys = {
        "embedding",
        "embedding1",
        "embedding2",
    }

    def _is_noisy_property(key: Any) -> bool:
        lk = str(key).strip().lower()
        return (not lk) or (lk in noisy_keys) or ("embed" in lk) or ("vector" in lk)

    def _clean_value(value: Any) -> Any:
        if isinstance(value, str) and len(value) > 400:
            return value[:397] + "..."
        if isinstance(value, list) and len(value) > 30:
            return value[:30]
        return value

    def _clean_props(props: Any) -> Dict[str, Any]:
        if not isinstance(props, dict):
            return {}
        return {k: _clean_value(v) for k, v in props.items() if not _is_noisy_property(k)}

    nodes_in = (merged_graph or {}).get("nodes", []) or []
    rels_in = (merged_graph or {}).get("relationships", []) or []

    cleaned_nodes: List[Dict[str, Any]] = []
    for node in nodes_in:
        if not isinstance(node, dict):
            continue
        cleaned_nodes.append(
            {
                "id": node.get("id"),
                "labels": node.get("labels", []),
                "properties": _clean_props(node.get("properties") or {}),
            }
        )

    if cypher_query:
        payload: Dict[str, Any] = {
            "cypher_query": cypher_query,
            "cypher_rows": (cypher_rows or [])[:30],
            "graph": {
                "nodes": cleaned_nodes,
                "relationships": rels_in,
            },
        }
    else:
        payload = {
            "nodes": cleaned_nodes,
            "relationships": rels_in,
        }

    text = json.dumps(payload, ensure_ascii=False, indent=2)

    if isinstance(max_chars, int) and max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def run_graphrag(
    query: str,
    retrieval_mode: str = "cypher",  # cypher | embeddings | hybrid
    llm_model: str = "meta-llama/llama-3.3-70b-instruct:free",
    use_llm_entities: bool = True,
    execute_cypher: bool = True,
    use_llm_judge: bool = False,
    judge_model: Optional[str] = None,
    embedding_top_k: int = 5,
    embedding_model: str = "model1",  # model1 | model2
) -> Dict[str, Any]:
    """High-level orchestrator returning a UI-friendly trace object."""
    retrieval_mode_norm = retrieval_mode.strip().lower()
    if retrieval_mode_norm not in {"cypher", "embeddings", "hybrid"}:
        retrieval_mode_norm = "cypher"

    cypher_trace: Optional[Dict[str, Any]] = None
    embedding_trace: Optional[Dict[str, Any]] = None
    llm_trace: Dict[str, Any] = {"model": llm_model, "answer": None, "error": None, "judge": None}

    if retrieval_mode_norm in {"cypher", "hybrid"}:
        cypher_result = process_query(
            query,
            use_llm=use_llm_entities,
            execute=execute_cypher,
            verbose=False,
        )
        cypher_trace = {
            "intent": cypher_result.get("intent"),
            "entities": cypher_result.get("entities"),
            "extraction_method": cypher_result.get("extraction_method"),
            "cypher_query": cypher_result.get("cypher_query"),
            "results": cypher_result.get("results"),
            "merged_graph": cypher_result.get("merged_graph"),
            "cypher_rows": cypher_result.get("cypher_rows"),
        }

    if retrieval_mode_norm in {"embeddings", "hybrid"}:
        try:
            # Lazy import to keep baseline Cypher mode lightweight.
            try:
                from . import embeddings as emb
            except ImportError:
                import embeddings as emb

            embedding_model_norm = (embedding_model or "model1").strip().lower()
            use_model2 = embedding_model_norm in {
                "model2",
                "embedding2",
                "gemma",
                "embeddinggemma",
                "embeddinggemma-300m",
                "google/embeddinggemma-300m",
            }

            if use_model2 and hasattr(emb, "get_model2"):
                model = emb.get_model2()
                embedding_property = "embedding2"
                embedding_model_name = "google/embeddinggemma-300m"
            else:
                model = emb.get_model1() if hasattr(emb, "get_model1") else emb.model1
                embedding_property = "embedding1"
                embedding_model_name = "thenlper/gte-small"

            embedding_result = emb.semantic_search(
                query,
                model,
                int(embedding_top_k),
                embedding_property=embedding_property,
            )
            # Normalize into the same node/relationship shape as Cypher.
            embedding_trace = {
                "top_k": int(embedding_top_k),
                "model": embedding_model_name,
                "embedding_property": embedding_property,
                "hits": embedding_result.get("nodes", []),
                "related_nodes": embedding_result.get("rel_nodes", []),
                "relationships": embedding_result.get("relationships", []),
            }
        except Exception as e:
            embedding_trace = {"error": str(e)}

    embedding_graph = _embeddings_trace_to_graph(embedding_trace)
    if embedding_trace is not None:
        embedding_trace["graph"] = embedding_graph

    retrieved_graph = _merge_graphs(
        [
            (cypher_trace or {}).get("merged_graph") if cypher_trace else None,
            embedding_graph,
        ]
    )

    # Build context text from the retrieved graph to keep the LLM input consistent.
    cypher_query_for_llm: Optional[str] = None
    if cypher_trace and cypher_trace.get("cypher_query"):
        cypher_query_for_llm = cypher_trace.get("cypher_query")

    cypher_rows_for_llm: Optional[List[Dict[str, Any]]] = None
    if cypher_trace and cypher_trace.get("cypher_rows"):
        cypher_rows_for_llm = cypher_trace.get("cypher_rows")

    context_text = _build_context_text(
        retrieved_graph,
        cypher_query=cypher_query_for_llm,
        cypher_rows=cypher_rows_for_llm,
    )

    try:
        try:
            from .llm import generate_answer
        except ImportError:
            from llm import generate_answer

        llm_answer, llm_metrics = generate_answer(
            prompt=query,
            context=context_text,
            model=llm_model,
            return_metrics=True,
        )
        llm_trace["answer"] = llm_answer
        llm_trace["metrics"] = llm_metrics
    except Exception as e:
        llm_trace["error"] = str(e)

    if use_llm_judge and llm_trace.get("answer") and not llm_trace.get("error"):
        try:
            try:
                from .llm import judge_answer
            except ImportError:
                from llm import judge_answer

            judge_model_name = (judge_model or llm_model).strip() if isinstance((judge_model or llm_model), str) else llm_model
            judge_result, judge_metrics = judge_answer(
                question=query,
                context=context_text,
                answer=str(llm_trace.get("answer") or ""),
                cypher_query=cypher_query_for_llm,
                model=judge_model_name,
                return_metrics=True,
            )
            llm_trace["judge"] = {
                "model": judge_model_name,
                "result": judge_result,
                "metrics": judge_metrics,
                "error": None,
            }
        except Exception as e:
            llm_trace["judge"] = {"model": judge_model or llm_model, "result": None, "metrics": None, "error": str(e)}

    return {
        "query": query,
        "retrieval_mode": retrieval_mode_norm,
        "cypher": cypher_trace,
        "embeddings": embedding_trace,
        "retrieved_graph": retrieved_graph,
        "context_text": context_text,
        "llm": llm_trace,
    }

def fill_template(template: str, entities: Dict) -> str:
    """
    Fill Cypher query template with extracted entities
    
    Args:
        template: Cypher query template with $placeholders
        entities: Extracted entities dictionary
    
    Returns:
        Filled Cypher query string
    """
    if not template:
        return ""
    
    filled = template

    # Capitalize passenger_class to match database format
    if 'passenger_class' in entities and isinstance(entities.get('passenger_class'), str):
        if entities['passenger_class'] not in ['all', 'INVALID', None]:
            entities['passenger_class'] = entities['passenger_class'].capitalize()

    if "fleet_type_description" in entities:
        entities["fleet_type"] = entities["fleet_type_description"]

    if "loyalty_program_level" in entities:
        entities["loyalty_level"] = entities["loyalty_program_level"]
    
    # Replace entity placeholders with actual values
    for key, value in entities.items():
        placeholder = f"${key}"
        if placeholder in filled:
            if value is None or value == 'all' or value == 'INVALID':
                # Skip INVALID values and let default handling take care of it
                continue
            elif isinstance(value, str):
                filled = filled.replace(placeholder, f"'{value}'")
            elif isinstance(value, (int, float)):
                filled = filled.replace(placeholder, str(value))
            elif isinstance(value, list) and value:
                filled = filled.replace(placeholder, f"'{value[0]}'")
    
    # Default values for remaining placeholders
    defaults = {
        '$limit': '10',
        '$min_satisfaction': '3',
        '$max_satisfaction': '3',
        '$max_delay': '30',
        '$min_legs': '2',
        '$fleet_type': "'all'",
        '$loyalty_level': "'all'",
        '$passenger_class': "'all'",
        '$origin_station_code': "NULL",
        '$destination_station_code': "NULL"
    }
    
    for placeholder, default in defaults.items():
        if placeholder in filled:
            filled = filled.replace(placeholder, default)
    
    return filled

if __name__ == "__main__":
    test_queries = [
        "What are the most popular routes?",
        "How satisfied are premier gold passengers?",
        "Which aircraft fleet types have the best satisfaction?",
        "Show insights for passengers with at least 3 legs", # hi
        "How satisfied are economy class passengers?", # not working
        "What's the performance of Boeing 737?",
        "Show me delayed flights",
    ]
    
    print("GRAPH-RAG TRAVEL ASSISTANT - INPUT PROCESSING PIPELINE")
    print("=" * 80)
    
    # Test with the first few queries and execute them
    for i, query in enumerate(test_queries[:3], 1):
        print(f"\n\n{'#' * 80}")
        print(f"TEST {i}/{3}")
        print(f"{'#' * 80}\n")
        
        result = process_query(query, use_llm=True, execute=True)
        
        print("\n" + "=" * 80)
        input("Press Enter to continue to next test...")
    
    print(f"\n\n{'#' * 80}")
    print("ALL TESTS COMPLETED")
    print(f"{'#' * 80}")
