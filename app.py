import os
from typing import Any, Dict, List

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from M3 import pipeline as graphrag

load_dotenv()

def _to_dataframe_nodes(nodes: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for node in nodes:
        props = node.get("properties", {}) or {}
        rows.append(
            {
                "id": node.get("id"),
                "labels": ",".join(node.get("labels", []) or []),
                **{f"prop.{k}": v for k, v in props.items()},
            }
        )
    return pd.DataFrame(rows)


def _to_dataframe_relationships(rels: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "type": r.get("type"),
                "start_node": r.get("start_node"),
                "end_node": r.get("end_node"),
            }
            for r in rels
        ]
    )


def _plot_graph(nodes: List[Dict[str, Any]], rels: List[Dict[str, Any]]):
    import networkx as nx
    import plotly.graph_objects as go

    g = nx.Graph()

    for node in nodes:
        node_id = node.get("id")
        labels = node.get("labels") or []
        props = node.get("properties") or {}
        title = labels[0] if labels else "Node"
        name = props.get("station_code") or props.get("flight_number") or props.get("loyalty_program_level")
        if name:
            title = f"{title}: {name}"
        g.add_node(node_id, title=title)

    for rel in rels:
        g.add_edge(rel.get("start_node"), rel.get("end_node"), label=rel.get("type"))

    if g.number_of_nodes() == 0:
        st.info("No nodes to visualize.")
        return

    pos = nx.spring_layout(g, seed=42)

    edge_x = []
    edge_y = []
    edge_text = []
    for u, v, data in g.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(data.get("label", ""))

    node_x = []
    node_y = []
    node_text = []
    for n, data in g.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(data.get("title", str(n)))

    fig = go.Figure(
        data=[
            go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=1),
                hoverinfo="none",
                mode="lines",
            ),
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=[t.split(":", 1)[0] for t in node_text],
                hovertext=node_text,
                hoverinfo="text",
                textposition="top center",
                marker=dict(size=10),
            ),
        ]
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=520,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _map_retrieval_mode(label: str) -> str:
    if label.startswith("Cypher"):
        return "cypher"
    if label.startswith("Embeddings"):
        return "embeddings"
    return "hybrid"


def _render_llm_judge(result: object) -> None:
    if not isinstance(result, dict):
        st.json(result)
        return

    # New qualitative schema (preferred)
    if isinstance(result.get("overall_verdict"), str) or isinstance(result.get("dimension_feedback"), list):
        overall_verdict = result.get("overall_verdict")
        if isinstance(overall_verdict, str) and overall_verdict.strip():
            st.markdown(f"**Overall verdict:** `{overall_verdict.strip()}`")

        strengths = result.get("top_strengths")
        improvements = result.get("top_improvements")
        if isinstance(strengths, list) or isinstance(improvements, list):
            left, right = st.columns(2)
            with left:
                if isinstance(strengths, list) and strengths:
                    st.markdown("**Top strengths**")
                    for item in strengths:
                        if item is None:
                            continue
                        st.markdown(f"- {str(item)}")
            with right:
                if isinstance(improvements, list) and improvements:
                    st.markdown("**Top improvements**")
                    for item in improvements:
                        if item is None:
                            continue
                        st.markdown(f"- {str(item)}")

        rewrite = result.get("rewrite_suggestion")
        if isinstance(rewrite, str) and rewrite.strip():
            st.markdown("**Rewrite suggestion**")
            st.write(rewrite.strip())

        dim_feedback = result.get("dimension_feedback")
        if isinstance(dim_feedback, list) and dim_feedback:
            st.markdown("**Rubric details**")
            for entry in dim_feedback:
                if not isinstance(entry, dict):
                    continue
                dimension = entry.get("dimension") or "Dimension"
                label = entry.get("label") or ""
                title = f"{dimension}" + (f" — {label}" if label else "")
                with st.expander(str(title), expanded=False):
                    evidence = entry.get("evidence")
                    if isinstance(evidence, list) and evidence:
                        st.markdown("**Evidence**")
                        for snippet in evidence[:3]:
                            if snippet is None:
                                continue
                            st.markdown(f"- {str(snippet)}")
                    commentary = entry.get("commentary")
                    if isinstance(commentary, str) and commentary.strip():
                        st.markdown("**Commentary**")
                        st.write(commentary.strip())

        with st.expander("Raw judge JSON", expanded=False):
            st.json(result)
        return

    # Legacy numeric schema fallback
    legacy_keys = {"groundedness", "completeness", "clarity", "overall", "rationale"}
    if legacy_keys.intersection(result.keys()):
        cols = st.columns(4)
        cols[0].metric("Groundedness", str(result.get("groundedness", "N/A")))
        cols[1].metric("Completeness", str(result.get("completeness", "N/A")))
        cols[2].metric("Clarity", str(result.get("clarity", "N/A")))
        cols[3].metric("Overall", str(result.get("overall", "N/A")))
        rationale = result.get("rationale")
        if isinstance(rationale, str) and rationale.strip():
            st.markdown("**Rationale**")
            st.write(rationale.strip())
        with st.expander("Raw judge JSON", expanded=False):
            st.json(result)
        return

    st.json(result)


st.set_page_config(page_title="Airline Graph-RAG Demo", layout="wide")

st.title("Airline Graph-RAG Demo")
st.caption("Graph-RAG with transparent KG retrieval → LLM answer")

with st.sidebar:
    st.subheader("Settings")

    llm_model = st.selectbox(
        "LLM model",
        options=[
            "meta-llama/llama-3.3-70b-instruct:free",
            "openai/gpt-oss-120b:free",
            "z-ai/glm-4.5-air:free",
        ],
        index=0,
    )

    use_llm_judge = st.checkbox("LLM judge (qualitative eval)", value=False)
    judge_model = llm_model
    if use_llm_judge:
        judge_model = st.selectbox(
            "Judge model",
            options=[
                llm_model,
                "meta-llama/llama-3.3-70b-instruct:free",
                "openai/gpt-oss-120b:free",
                "z-ai/glm-4.5-air:free",
            ],
            index=0,
        )

    retrieval_label = st.selectbox(
        "Retrieval method",
        options=[
            "Cypher (KG baseline)",
            "Embeddings",
            "Hybrid (Cypher + Embeddings)",
        ],
        index=0,
    )

    retrieval_mode = _map_retrieval_mode(retrieval_label)

    embedding_model = "model1"
    if retrieval_mode in {"embeddings", "hybrid"}:
        _embedding_model_choices = {
            "thenlper/gte-small (embedding1)": "model1",
            "google/embeddinggemma-300m (embedding2; requires HF_TOKEN)": "model2",
        }
        embedding_model_label = st.selectbox(
            "Embedding model",
            options=list(_embedding_model_choices.keys()),
            index=0,
        )
        embedding_model = _embedding_model_choices[embedding_model_label]

    use_llm_entities = st.checkbox("LLM entity extraction", value=True)
    execute_cypher = st.checkbox("Execute Cypher", value=True)
    embedding_top_k = st.slider(
        "Embedding top_k",
        min_value=1,
        max_value=20,
        value=5,
        disabled=retrieval_mode == "cypher",
    )

    show_graph_viz = st.checkbox("Show graph visualization", value=True)

    st.divider()
    st.subheader("Environment")
    st.text(f"NEO4J_URI: {'set' if os.getenv('NEO4J_URI') else 'not set'}")
    st.text(f"OPENROUTER_API_KEY: {'set' if os.getenv('OPENROUTER_API_KEY') else 'not set'}")

query = st.text_input(
    "Ask about routes, delays, satisfaction, fleets, etc.",
    placeholder="e.g., What are the top 5 busiest routes?",
)

run = st.button("Run", type="primary", disabled=not bool(query.strip()))

if run:
    with st.spinner("Running Graph-RAG..."):
        trace = graphrag.run_graphrag(
            query=query.strip(),
            retrieval_mode=retrieval_mode,
            llm_model=llm_model,
            use_llm_entities=use_llm_entities,
            execute_cypher=execute_cypher,
            use_llm_judge=use_llm_judge,
            judge_model=judge_model,
            embedding_top_k=int(embedding_top_k),
            embedding_model=embedding_model,
        )

    left, right = st.columns([1, 1])

    with left:
        st.subheader("KG-retrieved context (raw)")

        cy = trace.get("cypher") or {}
        retrieved = trace.get("retrieved_graph") or {}
        nodes = retrieved.get("nodes", []) if isinstance(retrieved, dict) else []
        rels = retrieved.get("relationships", []) if isinstance(retrieved, dict) else []

        st.markdown("**Intent & entities**")
        st.json(
            {
                "intent": cy.get("intent"),
                "extraction_method": cy.get("extraction_method"),
                "entities": cy.get("entities"),
            }
        )

        st.markdown("**Nodes**")
        st.dataframe(_to_dataframe_nodes(nodes), width="stretch")

        st.markdown("**Relationships**")
        st.dataframe(_to_dataframe_relationships(rels), width="stretch")

        if show_graph_viz:
            st.markdown("**Graph visualization**")
            _plot_graph(nodes, rels)

        st.subheader("Cypher queries executed")
        st.code(cy.get("cypher_query") or "(none)", language="cypher")

        emb = trace.get("embeddings")
        if emb is not None:
            st.subheader("Embedding retrieval (raw)")
            with st.expander("Embedding hits"):
                hits = emb.get("hits", []) if isinstance(emb, dict) else []
                if hits:
                    st.dataframe(pd.DataFrame(hits), width="stretch")
                else:
                    st.write("(none)")

            with st.expander("Embedding-derived graph"):
                graph = emb.get("graph") if isinstance(emb, dict) else None
                st.json(graph if graph else "(none)")

    with right:
        st.subheader("Final LLM answer")

        llm = trace.get("llm") or {}
        if llm.get("error"):
            st.error(f"LLM error: {llm.get('error')}")
            st.markdown("If you want LLM answers, set `OPENROUTER_API_KEY`.")
        else:
            st.write(llm.get("answer") or "")

        # Quantitative metrics: response time + token usage
        def _fmt_ms(value):
            return f"{value:.1f} ms" if isinstance(value, (int, float)) else "N/A"

        llm_metrics = llm.get("metrics") if isinstance(llm, dict) else None
        if isinstance(llm_metrics, dict):
            rt = llm_metrics.get("response_time_ms")
            tokens = (llm_metrics.get("tokens") or {}) if isinstance(llm_metrics.get("tokens"), dict) else {}
            total_tokens = tokens.get("total")
            st.markdown(
                f"**LLM metrics**  \
Response time: `{_fmt_ms(rt)}`" + (f"  \
Token usage (total): `{total_tokens}`" if total_tokens is not None else "  \
Token usage (total): `N/A`")
            )
        else:
            st.markdown("**LLM metrics**  \
Response time: `N/A`  \
Token usage (total): `N/A`")

        judge = llm.get("judge") if isinstance(llm, dict) else None
        if isinstance(judge, dict) and (judge.get("result") or judge.get("error")):
            st.markdown("**Qualitative (LLM judge)**")
            if judge.get("error"):
                st.error(f"Judge error: {judge.get('error')}")
            else:
                _render_llm_judge(judge.get("result"))

            judge_metrics = judge.get("metrics") if isinstance(judge.get("metrics"), dict) else None
            if isinstance(judge_metrics, dict):
                rt = judge_metrics.get("response_time_ms")
                tokens = (judge_metrics.get("tokens") or {}) if isinstance(judge_metrics.get("tokens"), dict) else {}
                total_tokens = tokens.get("total")
                st.markdown(
                    f"Judge response time: `{_fmt_ms(rt)}`" + (f"  \
Judge token usage (total): `{total_tokens}`" if total_tokens is not None else "  \
Judge token usage (total): `N/A`")
                )

        # Entity extraction metrics (only when LLM entity extraction is enabled)
        cy = trace.get("cypher") or {}
        cy_metrics = cy.get("metrics") if isinstance(cy, dict) else None
        ent_metrics = None
        if isinstance(cy_metrics, dict):
            ent_metrics = cy_metrics.get("entity_extraction")

        if isinstance(ent_metrics, dict):
            rt = ent_metrics.get("response_time_ms")
            tokens = (ent_metrics.get("tokens") or {}) if isinstance(ent_metrics.get("tokens"), dict) else {}
            total_tokens = tokens.get("total")
            st.markdown(
                f"**Entity-extraction LLM metrics**  \
Response time: `{_fmt_ms(rt)}`" + (f"  \
Token usage (total): `{total_tokens}`" if total_tokens is not None else "  \
Token usage (total): `N/A`")
            )

        with st.expander("Context passed to the LLM"):
            st.code(trace.get("context_text") or "", language="json")
