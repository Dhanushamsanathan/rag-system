"""
Streamlit Web UI Dashboard
Provides interactive Hybrid Search, RFP Q&A with citations, Bid Extraction Inspector,
and Retrieval Evaluation metrics dashboard.
"""

import os
import json
import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="RFP Intelligence Platform",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header styling (Dark and Light mode adaptive)
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; opacity: 0.8; margin-bottom: 1.5rem; }
    .citation-box {
        background-color: rgba(125, 125, 125, 0.12);
        border: 1px solid rgba(125, 125, 125, 0.25);
        border-left: 4px solid #3B82F6;
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📑 RFP Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Production RAG Hybrid Search Engine & Multi-Agent RFP Analysis System</div>', unsafe_allow_html=True)

# Lazy loading search engine components
@st.cache_resource
def load_system():
    from search.indexer import SearchIndexer
    from search.hybrid_retriever import HybridRetriever
    from agents.orchestrator import Orchestrator

    indexer = SearchIndexer()
    retriever = HybridRetriever(indexer=indexer)
    orchestrator = Orchestrator(indexer=indexer, retriever=retriever)
    return indexer, retriever, orchestrator

try:
    indexer, retriever, orchestrator = load_system()
    indexed_bids = indexer.list_indexed_bids()
except Exception as e:
    st.error(f"Error loading system backend: {e}")
    indexed_bids = ["Bid1", "Bid2"]

# Sidebar information
st.sidebar.title("Soliciation Controls")
selected_bid = st.sidebar.selectbox("Filter Target Bid", ["All Bids"] + indexed_bids)
target_bid_id = None if selected_bid == "All Bids" else selected_bid

st.sidebar.markdown("---")
st.sidebar.markdown("### System Architecture")
st.sidebar.markdown("""
- **Retrieval Engine**: Dense (BGE-Small) + Sparse (BM25Okapi)
- **Rank Fusion**: Reciprocal Rank Fusion ($k=60$)
- **Re-ranker**: BGE Cross-Encoder
- **Agent Roles**: Orchestrator, Ingestion, Retrieval, 3 Specialized Extractors, Addendum Reconciler, Critic Validator
""")

# Main Tabs
tab_search, tab_qa, tab_extraction, tab_eval, tab_compare = st.tabs([
    "🔍 Hybrid Search Engine",
    "💬 RFP Q&A Chatbot",
    "📊 Structured Extraction & Addenda",
    "📈 Retrieval Benchmark Evaluation",
    "⚖️ Bid Comparison & Go/No-Go"
])

# ----------------- TAB 1: HYBRID SEARCH -----------------
with tab_search:
    st.subheader("Dense & Sparse Hybrid Search with Re-ranking")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_query = st.text_input("Enter RFP search query:", value="warranty service repair turnaround SLA")
    with col2:
        search_mode = st.selectbox("Search Mode", ["hybrid", "dense", "sparse"])
    with col3:
        top_k_val = st.slider("Results (top_k)", min_value=1, max_value=10, value=5)

    doc_type_filter = st.selectbox("Metadata Filter: Document Type", ["Any", "rfp", "addendum", "specs", "affidavit", "bid_page"])
    filter_dict = {} if doc_type_filter == "Any" else {"doc_type": doc_type_filter}

    if st.button("Execute Search", type="primary"):
        with st.spinner("Searching indexed bid corpus..."):
            results = retriever.search(
                query=search_query,
                bid_id=target_bid_id,
                top_k=top_k_val,
                mode=search_mode,
                filters=filter_dict,
                apply_rerank=True,
                apply_expansion=True
            )

        if not results:
            st.warning("No matching passages found.")
        else:
            st.success(f"Retrieved {len(results)} ranked passages.")
            for idx, r in enumerate(results, 1):
                with st.expander(f"#{idx} | [{r.bid_id}] {r.file_name} (Page {r.page_number}) — Score: {r.score:.4f}", expanded=(idx <= 2)):
                    st.markdown(f"**Document Type:** `{r.doc_type}` | **Section:** `{r.section_title or 'General'}` | **Retrieval Method:** `{r.method}`")
                    st.code(r.text, language="markdown")

# ----------------- TAB 2: RFP Q&A -----------------
with tab_qa:
    st.subheader("Natural Language Q&A Grounded in Evidence")
    preset_questions = [
        "What is the submission deadline for Bid1 after all addendums?",
        "Which affidavits are required for the Dell laptop bid?",
        "Compare the warranty requirements of both bids.",
        "Is a bid bond required, and if so, how much?",
        "What changed in Addendum 2 compared to the original RFP?"
    ]
    selected_preset = st.selectbox("Choose a sample benchmark question or type below:", ["(Custom Question)"] + preset_questions)
    
    user_q = st.text_input("Question:", value=selected_preset if selected_preset != "(Custom Question)" else "")

    if st.button("Ask RFP Platform", type="primary"):
        if not user_q.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Consulting multi-agent Q&A coordinator..."):
                response = orchestrator.answer_question(user_q, bid_id=target_bid_id)

            st.markdown("### Answer")
            st.markdown(response.answer)

            st.markdown("### Traceable Evidence Citations")
            for idx, c in enumerate(response.citations, 1):
                with st.expander(f"📄 Source #{idx}: {c.file} — Page {c.page}", expanded=True):
                    st.markdown(f"**Document File:** `{c.file}` &nbsp;|&nbsp; **Page:** `{c.page}` &nbsp;|&nbsp; **Score:** `{c.score or 0.0:.4f}`")
                    st.info(f'"{c.text_snippet}"')

# ----------------- TAB 3: STRUCTURED EXTRACTION -----------------
with tab_extraction:
    st.subheader("Part D: 20-Field Structured Extraction Records")
    bid_choice = st.radio("Select Bid to inspect:", ["Bid1", "Bid2"], horizontal=True)
    out_file = f"output/{bid_choice.lower()}_extracted.json"

    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            data = json.load(f)

        # Validation badges
        val = data.get("validation", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fields Passed", val.get("passed", 0))
        c2.metric("Fields Failed", val.get("failed", 0))
        c3.metric("Not Found / Null", val.get("not_found", 0))
        c4.metric("Total Extracted", 20)

        # Display Addendum changes if present
        chgs = data.get("addendum_changes", [])
        if chgs:
            st.markdown("#### 🔄 Addendum Reconciliation Change Log")
            st.dataframe(pd.DataFrame(chgs), use_container_width=True)

        st.markdown("#### 📋 Field Specifications & Citations")
        fields = data.get("fields", {})
        table_rows = []
        for fname, fval in fields.items():
            src_str = ", ".join([f"{s.get('file', '')} (p.{s.get('page', '')})" for s in fval.get("sources", [])])
            table_rows.append({
                "Field Name": fname,
                "Extracted Value": str(fval.get("value", "null")),
                "Confidence": f"{fval.get('confidence', 0):.2f}",
                "Sources": src_str or "N/A",
                "Notes": fval.get("notes", "")
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        with st.expander("View Raw JSON Output"):
            st.json(data)
    else:
        st.info(f"Extraction file {out_file} not found. Run extraction via CLI or pipeline first.")

# ----------------- TAB 4: RETRIEVAL EVALUATION -----------------
with tab_eval:
    st.subheader("Section 6.4: Retrieval Evaluation Benchmark")
    eval_file = "output/retrieval_eval_report.json"
    if os.path.exists(eval_file):
        with open(eval_file, "r") as f:
            eval_data = json.load(f)

        st.markdown(f"**Benchmark Dataset:** `{eval_data.get('evaluation_pairs_count', 18)}` Question-Passage Ground Truth Pairs")
        configs = eval_data.get("configurations", [])
        df_eval = pd.DataFrame(configs)
        st.dataframe(df_eval, use_container_width=True)

        st.markdown("#### Key Benchmark Observations:")
        st.markdown("""
        1. **Hybrid + Cross-Encoder Re-ranker** yields the highest Recall@1 and MRR across both complex solicitation datasets.
        2. **Sparse BM25** is critical for exact solicitation identifier matching (e.g. `JA-207652`, `#E20P4600040`, `WD22TB4`, `210-BLYZ`).
        3. **Dense Vector Retrieval** provides conceptual matches for broad semantic questions (e.g. warranty SLAs, installation services, affidavits).
        """)
    else:
        st.info("Evaluation report not found. Run `python eval/evaluate_retrieval.py` first.")

# ----------------- TAB 5: BID COMPARISON & GO / NO-GO -----------------
with tab_compare:
    st.subheader("Bonus Features: Side-by-Side Comparison & Go/No-Go Decision Engine")

    comp_file = "output/bid_comparison_report.json"
    gonogo_file = "output/gonogo_report.json"

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Generate Fresh Comparative Analysis", type="primary"):
            from agents.comparison_agent import BidComparisonAgent
            from agents.gonogo_agent import GoNoGoAgent
            with open("output/bid1_extracted.json") as f:
                b1 = json.load(f)
            with open("output/bid2_extracted.json") as f:
                b2 = json.load(f)
            comparator = BidComparisonAgent()
            rep = comparator.compare_bids([b1, b2])
            with open(comp_file, "w") as f:
                json.dump(rep.model_dump(), f, indent=2)

            gonogo = GoNoGoAgent()
            g1 = gonogo.evaluate_bid(b1)
            g2 = gonogo.evaluate_bid(b2)
            with open(gonogo_file, "w") as f:
                json.dump({"Bid1": g1.model_dump(), "Bid2": g2.model_dump()}, f, indent=2)
            st.success("Comparative analysis and Go/No-Go recommendations updated!")

    # Display Go / No-Go Recommendations
    st.markdown("### 🚦 Automated Pursuit Recommendation (Go / No-Go)")
    if os.path.exists(gonogo_file):
        with open(gonogo_file) as f:
            gdata = json.load(f)

        g_col1, g_col2 = st.columns(2)
        for idx, (bid_key, rec) in enumerate(gdata.items()):
            col = g_col1 if idx == 0 else g_col2
            with col:
                decision = rec.get("decision", "REVIEW")
                score = rec.get("match_score", 0.0)
                badge_color = "#10B981" if decision == "GO" else ("#F59E0B" if decision == "CONDITIONAL_GO" else "#EF4444")
                st.markdown(f"""
                <div style="border: 1px solid rgba(125,125,125,0.3); border-radius: 8px; padding: 1rem; border-top: 5px solid {badge_color};">
                    <h4>{bid_key} &nbsp; <span style="background-color: {badge_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.9rem;">{decision}</span></h4>
                    <p><b>Pursuit Viability Score:</b> <code>{score}/100</code></p>
                    <p><i>{rec.get('executive_summary', '')}</i></p>
                    <b>Key Operational Strengths:</b>
                    <ul>{"".join([f"<li>{s}</li>" for s in rec.get('strengths', [])])}</ul>
                    <b>Identified Blockers / Items:</b>
                    <ul>{"".join([f"<li>{r}</li>" for r in rec.get('risks_and_blockers', [])])}</ul>
                </div>
                """, unsafe_allow_html=True)

    # Display Side-by-Side Matrix
    st.markdown("### ⚖️ Side-by-Side RFP Comparison Matrix")
    if os.path.exists(comp_file):
        with open(comp_file) as f:
            cdata = json.load(f)

        st.dataframe(pd.DataFrame(cdata.get("matrix", [])), use_container_width=True)

        st.markdown("#### Strategic Insights:")
        for aspect, narrative in cdata.get("comparative_analysis", {}).items():
            st.markdown(f"- **{aspect}:** {narrative}")

