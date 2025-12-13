"""
Streamlit UI for Project Green Lattern

Interactive interface for SEC filings analysis with:
- Document selection
- Analysis results visualization
- Citation tracking
- Interactive charts
- Comparison views
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import config
from agents import (
    SummaryAgentV2,
    SWOTAgentV2,
    MetricsAgentV2,
    DecisionAgentV2
)

# Page configuration
st.set_page_config(
    page_title="Project Green Lattern - SEC Analysis",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Fixed for dark mode compatibility
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        color: #000000;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card h2, .metric-card h3, .metric-card h4, .metric-card p {
        color: #000000 !important;
    }
    .citation {
        background-color: #e3f2fd;
        color: #000000;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .red-flag {
        background-color: #ffebee;
        color: #000000;
        padding: 0.5rem;
        border-left: 4px solid #f44336;
        margin: 0.5rem 0;
    }
    .green-flag {
        background-color: #e8f5e9;
        color: #000000;
        padding: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Utility functions
@st.cache_data
def load_available_documents() -> List[Dict]:
    """Load list of processed documents."""
    docs = []
    
    if not config.paths.final_dir.exists():
        return docs
    
    for doc_dir in config.paths.final_dir.iterdir():
        if doc_dir.is_dir():
            manifest_path = config.paths.raw_dir / doc_dir.name / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    docs.append({
                        "doc_id": doc_dir.name,
                        "ticker": manifest.get('ticker', 'Unknown'),
                        "form_type": manifest.get('form_type', 'Unknown'),
                        "filing_date": manifest.get('filing_date', 'Unknown')
                    })
    
    return sorted(docs, key=lambda x: x['filing_date'], reverse=True)


@st.cache_data
def load_analysis(doc_id: str, analysis_type: str) -> Optional[Dict]:
    """Load specific analysis results."""
    analysis_path = config.paths.final_dir / doc_id / f"{analysis_type}.json"
    
    if not analysis_path.exists():
        return None
    
    with open(analysis_path, 'r') as f:
        return json.load(f)


def format_citation(citation: Dict) -> str:
    """Format citation for display."""
    section = citation.get('section', '')
    page = citation.get('page', 0)
    method = citation.get('extraction_method', 'unknown')
    
    section_str = f"{section}, " if section else ""
    return f"[{section_str}Page {page + 1}, via {method}]"


def display_metric_card(title: str, value: str, delta: Optional[str] = None):
    """Display a metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin: 0; color: #1f77b4 !important;">{title}</h3>
        <h2 style="margin: 0.5rem 0; color: #000000 !important;">{value}</h2>
        {f'<p style="margin: 0; color: #333333 !important;">{delta}</p>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)


# Main app
def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🏢 Project Green Lattern</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">Open-Source SEC Filings Analysis with Multi-Agent Investment Intelligence</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Green+Lattern", use_container_width=True)
        
        st.header("📁 Document Selection")
        
        # Load available documents
        docs = load_available_documents()
        
        if not docs:
            st.warning("No processed documents found. Please run the pipeline first.")
            st.code("python run_pipeline.py --ticker AAPL --form-type 10-K")
            st.stop()
        
        # Document selector
        doc_options = [f"{d['ticker']} - {d['form_type']} ({d['filing_date']})" for d in docs]
        selected_idx = st.selectbox(
            "Select Document",
            range(len(doc_options)),
            format_func=lambda i: doc_options[i]
        )
        
        selected_doc = docs[selected_idx]
        doc_id = selected_doc['doc_id']
        
        st.success(f"Selected: {selected_doc['ticker']} {selected_doc['form_type']}")
        
        st.markdown("---")
        
        # Analysis options
        st.header("⚙️ Analysis Options")
        
        analysis_types = st.multiselect(
            "Select Analyses to Display",
            ["Summary", "SWOT", "Metrics", "Decision"],
            default=["Summary", "SWOT", "Metrics", "Decision"]
        )
        
        show_citations = st.checkbox("Show Citations", value=True)
        show_provenance = st.checkbox("Show Detailed Provenance", value=False)
        
        st.markdown("---")
        
        # Actions
        st.header("🚀 Actions")
        
        if st.button("🔄 Re-run All Analyses", use_container_width=True):
            with st.spinner("Running analyses..."):
                try:
                    # Run agents
                    summary_agent = SummaryAgentV2()
                    summary_agent.analyze(doc_id)
                    
                    swot_agent = SWOTAgentV2()
                    swot_agent.analyze(doc_id)
                    
                    metrics_agent = MetricsAgentV2()
                    metrics_agent.analyze(doc_id)
                    
                    decision_agent = DecisionAgentV2()
                    decision_agent.analyze(doc_id)
                    
                    st.success("✅ All analyses completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error running analyses: {e}")
        
        if st.button("📊 Export Report", use_container_width=True):
            st.info("Export functionality coming soon!")
    
    # Main content area
    tabs = st.tabs(["📋 Overview", "📝 Summary", "🎯 SWOT", "💰 Metrics", "🎲 Decision", "📚 Citations"])
    
    # Tab 1: Overview
    with tabs[0]:
        display_overview(doc_id, selected_doc)
    
    # Tab 2: Summary
    with tabs[1]:
        if "Summary" in analysis_types:
            display_summary(doc_id, show_citations, show_provenance)
        else:
            st.info("Summary analysis not selected")
    
    # Tab 3: SWOT
    with tabs[2]:
        if "SWOT" in analysis_types:
            display_swot(doc_id, show_citations, show_provenance)
        else:
            st.info("SWOT analysis not selected")
    
    # Tab 4: Metrics
    with tabs[3]:
        if "Metrics" in analysis_types:
            display_metrics(doc_id, show_citations)
        else:
            st.info("Metrics analysis not selected")
    
    # Tab 5: Decision
    with tabs[4]:
        if "Decision" in analysis_types:
            display_decision(doc_id)
        else:
            st.info("Decision analysis not selected")
    
    # Tab 6: Citations
    with tabs[5]:
        display_citations(doc_id)


def display_overview(doc_id: str, doc_info: Dict):
    """Display overview dashboard."""
    st.markdown('<h2 class="sub-header">📊 Analysis Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #000000 !important;">Company</h4>
            <h2 style="color: #000000 !important;">{doc_info['ticker']}</h2>
            <p style="color: #000000 !important;">Filing Type: {doc_info['form_type']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #000000 !important;">Filing Date</h4>
            <h2 style="color: #000000 !important;">{doc_info['filing_date']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Load decision for quick rating
        decision = load_analysis(doc_id, "decision_analysis_v2")
        if decision:
            rating = decision['investment_memo']['recommendation']['rating']
            color = {
                "Strong Buy": "#4caf50",
                "Buy": "#8bc34a",
                "Hold": "#ff9800",
                "Sell": "#ff5722",
                "Strong Sell": "#f44336"
            }.get(rating, "#666")
            
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {color};">
                <h4 style="color: #000000 !important;">Recommendation</h4>
                <h2 style="color: {color} !important;">{rating}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📈 Analysis Status")
        
        analyses = {
            "Summary": load_analysis(doc_id, "summary_analysis_v2"),
            "SWOT": load_analysis(doc_id, "swot_analysis_v2"),
            "Metrics": load_analysis(doc_id, "metrics_analysis_v2"),
            "Decision": load_analysis(doc_id, "decision_analysis_v2")
        }
        
        for name, data in analyses.items():
            status = "✅ Complete" if data else "❌ Not Run"
            st.markdown(f"**{name}:** {status}")
    
    with col2:
        st.markdown("### 🎯 Quick Insights")
        
        decision = load_analysis(doc_id, "decision_analysis_v2")
        if decision:
            memo = decision['investment_memo']
            
            st.markdown(f"**Quality Score:** {memo['quality_score']['score']}/100")
            st.markdown(f"**Balance Sheet Risk:** {memo['balance_sheet_risk']['risk_level']}")
            st.markdown(f"**Earnings Quality:** {memo['earnings_quality']['quality']}")
            st.markdown(f"**Red Flags:** {len(memo['red_flags'])}")


def display_summary(doc_id: str, show_citations: bool, show_provenance: bool):
    """Display summary analysis."""
    st.markdown('<h2 class="sub-header">📝 Executive Brief</h2>', unsafe_allow_html=True)
    
    summary = load_analysis(doc_id, "summary_analysis_v2")
    
    if not summary:
        st.warning("Summary analysis not available. Please run the analysis first.")
        return
    
    brief = summary['executive_brief']
    provenance_data = summary.get('provenance', [])
    
    # Key Findings
    st.markdown("### 📋 Key Findings")
    for i, finding in enumerate(brief['key_findings'], 1):
        with st.expander(f"Finding {i}: {finding['section']}", expanded=i <= 3):
            st.markdown(finding['finding'])
            if show_citations:
                st.caption(f"📍 Page {finding['page'] + 1} | {finding['section']}")
    
    # Company Direction
    st.markdown("### 🎯 Company Direction")
    st.info(brief['company_direction']['text'])
    
    # Delta Analysis
    if brief['delta_analysis'].get('available'):
        st.markdown("### 📊 What Changed vs Last Year")
        for change in brief['delta_analysis']['changes']:
            st.markdown(f"• {change}")
    
    # Financial Highlights
    if brief['financial_highlights'].get('available'):
        st.markdown("### 💰 Financial Highlights")
        
        cols = st.columns(len(brief['financial_highlights']['metrics']))
        for i, metric in enumerate(brief['financial_highlights']['metrics']):
            with cols[i]:
                display_metric_card(
                    metric['metric'],
                    f"{metric['value']:,.0f}",
                    metric.get('units', '')
                )
    
    # New Risks
    st.markdown("### ⚠️ Key Risk Factors")
    for risk in brief['new_risks']['risks']:
        st.markdown(f"""
        <div class="red-flag" style="color: #000000 !important;">
            {risk}
        </div>
        """, unsafe_allow_html=True)


def display_swot(doc_id: str, show_citations: bool, show_provenance: bool):
    """Display SWOT analysis."""
    st.markdown('<h2 class="sub-header">🎯 SWOT Analysis (Hostile Witness Mode)</h2>', unsafe_allow_html=True)
    
    swot = load_analysis(doc_id, "swot_analysis_v2")
    
    if not swot:
        st.warning("SWOT analysis not available.")
        return
    
    analysis = swot['swot_analysis']
    
    # Create 2x2 grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💪 Strengths")
        for item in analysis['strengths']['items']:
            with st.expander(f"{item['category']}: {item['strength'][:50]}..."):
                st.markdown(f"**Strength:** {item['strength']}")
                st.markdown(f"**Evidence:** {item['evidence']}")
                st.markdown(f"**Significance:** {item['significance']}")
                if show_citations:
                    st.caption(f"📍 Page {item['page'] + 1}")
    
    with col2:
        st.markdown("### ⚠️ Weaknesses")
        for item in analysis['weaknesses']['items']:
            with st.expander(f"{item['category']}: {item['weakness'][:50]}..."):
                st.markdown(f"**Weakness:** {item['weakness']}")
                st.markdown(f"**Evidence:** {item['evidence']}")
                st.markdown(f"**Severity:** {item['severity']}")
                if show_citations:
                    st.caption(f"📍 Page {item['page'] + 1}")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 🚀 Opportunities")
        for item in analysis['opportunities']['items']:
            with st.expander(f"{item['category']}: {item['opportunity'][:50]}..."):
                st.markdown(f"**Opportunity:** {item['opportunity']}")
                st.markdown(f"**Evidence:** {item['evidence']}")
                st.markdown(f"**Potential:** {item['potential']}")
                if show_citations:
                    st.caption(f"📍 Page {item['page'] + 1}")
    
    with col4:
        st.markdown("### 🔴 Threats")
        for item in analysis['threats']['items']:
            with st.expander(f"{item['category']}: {item['threat'][:50]}..."):
                st.markdown(f"**Threat:** {item['threat']}")
                st.markdown(f"**Evidence:** {item['evidence']}")
                st.markdown(f"**Severity:** {item['severity']}")
                if show_citations:
                    st.caption(f"📍 Page {item['page'] + 1}")
    
    # Risk Factor Delta
    if swot.get('risk_factor_delta', {}).get('available'):
        st.markdown("---")
        st.markdown("### 📊 Risk Factor Delta (Item 1A YoY)")
        
        delta = swot['risk_factor_delta']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Added Risks:**")
            for risk in delta.get('added_risks', []):
                st.markdown(f"• {risk['risk']}")
        
        with col2:
            st.markdown("**Heightened Risks:**")
            for risk in delta.get('heightened_risks', []):
                st.markdown(f"• {risk['risk']}")


def display_metrics(doc_id: str, show_citations: bool):
    """Display financial metrics."""
    st.markdown('<h2 class="sub-header">💰 Financial Metrics</h2>', unsafe_allow_html=True)
    
    metrics = load_analysis(doc_id, "metrics_analysis_v2")
    
    if not metrics:
        st.warning("Metrics analysis not available.")
        return
    
    m = metrics['metrics']
    
    # Profitability
    st.markdown("### 📈 Profitability Ratios")
    
    if m.get('profitability'):
        prof_data = []
        for name, data in m['profitability'].items():
            prof_data.append({
                "Ratio": name.replace('_', ' ').title(),
                "Value": f"{data['value']:.2f}{data['units']}",
                "Formula": data['formula']
            })
        
        df = pd.DataFrame(prof_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Visualization
        fig = go.Figure(data=[
            go.Bar(
                x=[d['Ratio'] for d in prof_data],
                y=[float(d['Value'].rstrip('%')) for d in prof_data],
                marker_color='#1f77b4'
            )
        ])
        fig.update_layout(
            title="Profitability Margins",
            xaxis_title="Metric",
            yaxis_title="Percentage (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Liquidity
    st.markdown("### 🏦 Liquidity & Solvency")
    
    if m.get('liquidity'):
        liq_cols = st.columns(len(m['liquidity']))
        for i, (name, data) in enumerate(m['liquidity'].items()):
            with liq_cols[i]:
                display_metric_card(
                    name.replace('_', ' ').title(),
                    f"{data['value']:.2f}{data['units']}",
                    data['formula']
                )
    
    # Cash Flow
    st.markdown("### 💵 Cash Flow Analysis")
    
    if m.get('cash_flow_analysis'):
        cf_data = []
        for name, data in m['cash_flow_analysis'].items():
            cf_data.append({
                "Metric": name.replace('_', ' ').title(),
                "Value": f"{data['value']:.2f}{data['units']}",
                "Formula": data['formula']
            })
        
        df = pd.DataFrame(cf_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


def display_decision(doc_id: str):
    """Display investment decision."""
    st.markdown('<h2 class="sub-header">🎲 Investment Memo</h2>', unsafe_allow_html=True)
    
    decision = load_analysis(doc_id, "decision_analysis_v2")
    
    if not decision:
        st.warning("Decision analysis not available.")
        return
    
    memo = decision['investment_memo']
    rec = memo['recommendation']
    
    # Recommendation
    rating_colors = {
        "Strong Buy": "#4caf50",
        "Buy": "#8bc34a",
        "Hold": "#ff9800",
        "Sell": "#ff5722",
        "Strong Sell": "#f44336"
    }
    
    color = rating_colors.get(rec['rating'], "#666")
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}22 0%, {color}11 100%); 
                padding: 2rem; border-radius: 15px; border-left: 8px solid {color};">
        <h1 style="color: {color} !important; margin: 0;">{rec['rating']}</h1>
        <p style="font-size: 1.2rem; margin: 0.5rem 0; color: #000000 !important;">Confidence: {rec['confidence']}</p>
        <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0; color: #000000 !important;">
            Composite Score: {rec['composite_score']}/100
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Scores
    col1, col2, col3 = st.columns(3)
    
    with col1:
        qs = memo['quality_score']
        display_metric_card(
            "Quality Score",
            f"{qs['score']}/100",
            qs['rating']
        )
    
    with col2:
        bsr = memo['balance_sheet_risk']
        display_metric_card(
            "Balance Sheet Risk",
            bsr['risk_level'],
            f"Risk Score: {bsr['risk_score']}"
        )
    
    with col3:
        eq = memo['earnings_quality']
        display_metric_card(
            "Earnings Quality",
            eq['quality'],
            f"{eq['num_flags']} red flags"
        )
    
    # Thesis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Bull Thesis")
        for point in memo['bull_thesis']:
            st.markdown(f"""
            <div class="green-flag" style="color: #000000 !important;">
                {point}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📉 Bear Thesis")
        for point in memo['bear_thesis']:
            st.markdown(f"""
            <div class="red-flag" style="color: #000000 !important;">
                {point}
            </div>
            """, unsafe_allow_html=True)
    
    # Red Flags
    if memo['red_flags']:
        st.markdown("### 🚩 Red Flags")
        for flag in memo['red_flags']:
            st.markdown(f"""
            <div class="red-flag" style="color: #000000 !important;">
                ⚠️ {flag}
            </div>
            """, unsafe_allow_html=True)
    
    # Monitoring Plan
    st.markdown("### 👀 What to Monitor")
    for item in memo['monitoring_plan']:
        st.markdown(f"• {item}")


def display_citations(doc_id: str):
    """Display all citations."""
    st.markdown('<h2 class="sub-header">📚 Citations & Provenance</h2>', unsafe_allow_html=True)
    
    # Load all analyses
    analyses = {
        "Summary": load_analysis(doc_id, "summary_analysis_v2"),
        "SWOT": load_analysis(doc_id, "swot_analysis_v2"),
        "Metrics": load_analysis(doc_id, "metrics_analysis_v2"),
        "Decision": load_analysis(doc_id, "decision_analysis_v2")
    }
    
    all_citations = []
    
    for name, data in analyses.items():
        if data and 'provenance' in data:
            for citation in data['provenance']:
                citation['source_analysis'] = name
                all_citations.append(citation)
    
    if not all_citations:
        st.info("No citations available.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_citations)
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Citations", len(all_citations))
    
    with col2:
        if 'page' in df.columns:
            st.metric("Pages Cited", df['page'].nunique())
    
    with col3:
        if 'extraction_method' in df.columns:
            st.metric("Extraction Methods", df['extraction_method'].nunique())
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        if 'source_analysis' in df.columns:
            analysis_filter = st.multiselect(
                "Filter by Analysis",
                df['source_analysis'].unique(),
                default=df['source_analysis'].unique()
            )
            df = df[df['source_analysis'].isin(analysis_filter)]
    
    with col2:
        if 'section' in df.columns:
            section_filter = st.multiselect(
                "Filter by Section",
                df['section'].dropna().unique() if 'section' in df.columns else []
            )
            if section_filter:
                df = df[df['section'].isin(section_filter)]
    
    # Display citations table
    st.dataframe(
        df[['text', 'page', 'section', 'extraction_method', 'source_analysis']],
        use_container_width=True,
        hide_index=True
    )


if __name__ == "__main__":
    main()
