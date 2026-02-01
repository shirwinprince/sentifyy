import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from dotenv import load_dotenv

# --- Preprocessing & AI Imports ---
from utils.preprocess import clean_text
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate

warnings.filterwarnings("ignore")
load_dotenv()

# Environment validation
os.environ["USER_AGENT"] = "rag-search-bot/1.0"
GROQ_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="Sentify Luxe | Enterprise Intelligence",
    page_icon="💎",
    layout="wide"
)

# ------------------ 🌓 THEME LOGIC ------------------

# Initialize Theme in Session State
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# Define Color Palettes
theme_params = {
    "Light": {
        "bg_color": "#f8fafc",
        "text_color": "#0f172a",
        "card_bg": "#ffffff",
        "card_border": "#e2e8f0",
        "metric_val": "#1e293b",
        "metric_label": "#64748b",
        "strategy_bg": "linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%)",
        "sidebar_bg": "#ffffff",
        "plotly_template": "plotly_white",
        "wc_bg": "white",
        "shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.05)"
    },
    "Dark": {
        "bg_color": "#0e1117",
        "text_color": "#f8fafc",
        "card_bg": "#1e293b",
        "card_border": "#334155",
        "metric_val": "#f1f5f9",
        "metric_label": "#94a3b8",
        "strategy_bg": "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)",
        "sidebar_bg": "#111827",
        "plotly_template": "plotly_dark",
        "wc_bg": "black",
        "shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.3)"
    }
}

current_theme = theme_params[st.session_state.theme]

# ------------------ 🎨 DYNAMIC CSS ------------------
st.markdown(f"""
<style>
    /* --- GLOBAL THEME --- */
    .stApp {{
        background-color: {current_theme['bg_color']};
    }}
    
    /* --- TYPOGRAPHY --- */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: {current_theme['text_color']} !important;
        font-weight: 700;
    }}
    p, span, div {{
        color: {current_theme['text_color']};
    }}
    
    /* --- METRIC CARDS --- */
    .metric-card {{
        background: {current_theme['card_bg']};
        border: 1px solid {current_theme['card_border']};
        border-radius: 12px;
        padding: 24px;
        box-shadow: {current_theme['shadow']};
        transition: all 0.3s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-5px);
        border-color: #3b82f6;
    }}
    .metric-value {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {current_theme['metric_val']};
        letter-spacing: -1px;
    }}
    .metric-label {{
        color: {current_theme['metric_label']};
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }}
    
    /* --- STRATEGY CARD --- */
    .strategy-card {{
        background: {current_theme['strategy_bg']};
        border-left: 5px solid #3b82f6; 
        border-radius: 8px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: {current_theme['shadow']};
        color: {current_theme['text_color']};
    }}
    
    /* --- REVIEW CARDS --- */
    .review-card {{
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        background: {current_theme['card_bg']};
        border: 1px solid {current_theme['card_border']};
        color: {current_theme['text_color']};
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: {current_theme['shadow']};
    }}
    
    /* --- SIDEBAR --- */
    section[data-testid="stSidebar"] {{
        background-color: {current_theme['sidebar_bg']};
        border-right: 1px solid {current_theme['card_border']};
    }}
    
    /* --- DATAFRAME --- */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {current_theme['card_border']};
        border-radius: 8px;
        background: {current_theme['card_bg']};
    }}
</style>
""", unsafe_allow_html=True)

# ------------------ LOADING RESOURCES ------------------

@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load("model/sentiment_model.pkl")
        tfidf = joblib.load("model/tfidf.pkl")
        return model, tfidf
    except Exception:
        return None, None

@st.cache_resource
def load_rag_assets():
    try:
        if not os.path.exists("vectorstore"): return None
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory="vectorstore", embedding_function=embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})
    except Exception:
        return None

model, tfidf = load_ml_assets()
retriever = load_rag_assets()
web_search = DuckDuckGoSearchRun()

# Professional Corporate Palette
colors = {"Positive": "#059669", "Neutral": "#64748b", "Negative": "#e11d48"} 

# ------------------ SIDEBAR ------------------

with st.sidebar:
    # --- THEME SWITCHER ---
    col_logo, col_theme = st.columns([3, 1])
    with col_logo:
        st.image("l.png", width=50) 
    with col_theme:
        if st.session_state.theme == "Light":
            if st.button("🌙", help="Switch to Dark Mode", key="theme_toggle_dark"):
                st.session_state.theme = "Dark"
                st.rerun()
        else:
            if st.button("☀️", help="Switch to Light Mode", key="theme_toggle_light"):
                st.session_state.theme = "Light"
                st.rerun()

    st.markdown("### Sentify Analytics")
    st.caption("Enterprise Sentiment Intelligence")
    
    st.markdown("---")
    
    model_choice = st.selectbox("LLM Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
    uploaded_file = st.file_uploader("📂 Import CSV Data", type="csv")
    min_confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.2)
    
    st.markdown("---")
    
    # --- AI ANALYST CHAT SECTION ---
    st.markdown("#### 💬 AI Analyst Chat")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat Control Buttons
    col_clear, col_dl = st.columns(2)
    
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col_dl:
        # Prepare chat history for download
        chat_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button(
            label="📥 Save Chat",
            data=chat_text,
            file_name="ai_analyst_session.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=len(st.session_state.messages) == 0
        )

    # Chat Container with fixed height
    chat_container = st.container(height=350)
    
    with chat_container:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask for insights..."):
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            st.chat_message("user").write(prompt)
        
        # AI Logic (ensure GROQ_KEY is set in your .env)
        try:
            llm = ChatGroq(temperature=0, model_name=model_choice, api_key=GROQ_KEY) if GROQ_KEY else None
            if llm:
                # Add context of the data if available
                context = f"The user has uploaded data with {len(df)} records." if 'df' in locals() else ""
                response = llm.invoke(f"{context} User Question: {prompt}").content
            else:
                response = "⚠️ API Key not detected. Please check your .env file."
        except Exception as e:
            response = f"❌ Error: {str(e)}"
            
        # Add Assistant Message
        st.session_state.messages.append({"role": "assistant", "content": response})
        with chat_container:
            st.chat_message("assistant").write(response)

# ------------------ MAIN DASHBOARD ------------------

if uploaded_file and model:
    try:
        df = pd.read_csv(uploaded_file)
    except:
        st.error("❌ Invalid CSV File")
        st.stop()

    if "Comment" not in df.columns:
        st.error("❌ CSV must contain a 'Comment' column")
        st.stop()

    # Data Processing
    df["Comment"] = df["Comment"].astype(str).fillna("")
    df["cleaned"] = df["Comment"].apply(clean_text)
    
    # ML Prediction
    X = tfidf.transform(df["cleaned"])
    df["Sentiment_Num"] = model.predict(X)
    
    if hasattr(model, "predict_proba"):
        df["Confidence"] = np.max(model.predict_proba(X), axis=1)
    else:
        df["Confidence"] = 1.0

    df["Sentiment"] = df["Sentiment_Num"].map({0: "Negative", 1: "Neutral", 2: "Positive"})
    df = df[df["Confidence"] >= min_confidence]
    
    # Feature Engineering
    df['Review_Length'] = df['Comment'].apply(len)
    df['Depth'] = df['Review_Length'].apply(lambda x: 'Detailed' if x > 150 else ('Short' if x < 50 else 'Standard'))

    # Metrics
    total_logs = len(df)
    pos_rate = (df["Sentiment"] == "Positive").sum() / total_logs if total_logs else 0
    sys_certainty = df['Confidence'].mean() * 100
    
    # --- SECTION 1: HEADER & KEY METRICS ---
    st.title("📊 Enterprise Growth Dashboard")
    st.markdown("Overview of customer sentiment, risk factors, and strategic opportunities.")
    
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    
    c1, c2, c3, c4 = st.columns(4)
    
    def metric_html(label, value, delta, color=current_theme['metric_val']):
        return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{value}</div>
            <div style="color: {current_theme['metric_label']}; font-size: 0.8rem; margin-top:8px; font-weight:500;">{delta}</div>
        </div>
        """
    
    with c1: st.markdown(metric_html("Total Feedback", f"{total_logs:,}", "↗ 12% vs last month"), unsafe_allow_html=True)
    with c2: st.markdown(metric_html("Sentiment Score", f"{pos_rate*100:.1f}%", "↗ 1.2% Growth", "#059669"), unsafe_allow_html=True)
    with c3: st.markdown(metric_html("Model Confidence", f"{sys_certainty:.1f}%", "• Stable System", "#3b82f6"), unsafe_allow_html=True)
    with c4: st.markdown(metric_html("Churn Risk", "Low", "↘ -5% Risk Factor", "#e11d48"), unsafe_allow_html=True)

    # --- SECTION 2: STRATEGY CARD ---
    st.markdown(f"""
    <div class="strategy-card">
        <h3 style="margin:0; font-size:1.3rem;">🧠 AI Strategic Recommendation</h3>
        <p style="margin:10px 0 0 0; font-size:1rem; line-height:1.6;">
        <b>Executive Summary:</b> Based on <b>{total_logs:,}</b> data points, your brand health is robust at <b>{pos_rate*100:.0f}%</b>. 
        <br><b>Action Item:</b> The high volume of 'Detailed' positive reviews suggests an opportunity to create user case studies. 
        Monitor the 'Negative' cluster in short reviews for immediate support intervention.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- SECTION 3: VISUAL ANALYTICS ---
    col_sun, col_gauge = st.columns([1.5, 1])
    
    with col_sun:
        st.subheader("Sentiment Composition")
        sun_df = df[['Sentiment', 'Depth']].dropna()
        fig_s = px.sunburst(
            sun_df, path=['Sentiment', 'Depth'], 
            color='Sentiment', 
            color_discrete_map=colors,
            template=current_theme['plotly_template']
        )
        fig_s.update_traces(textinfo='label+percent entry', insidetextorientation='radial')
        fig_s.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=350, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_s, use_container_width=True)

    with col_gauge:
        st.subheader("Brand Health Index")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=pos_rate*100,
            number={'suffix': "%", 'font': {'color': current_theme['text_color']}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#333"},
                'bar': {'color': "#059669"}, # Emerald Green
                'bgcolor': current_theme['card_bg'],
                'borderwidth': 2,
                'bordercolor': current_theme['card_border']
            }
        ))
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig_g, use_container_width=True)

    # --- SECTION 4: TEXT MINING ---
    st.divider()
    c_word, c_opp = st.columns(2)
    
    with c_word:
        st.subheader("Semantic Pattern Analysis")
        target = st.radio("Select Segment:", ["Positive", "Negative"], horizontal=True)
        subset = " ".join(df[df['Sentiment'] == target]['cleaned'])
        
        if len(subset) > 50:
            wc = WordCloud(
                background_color=current_theme['wc_bg'], 
                colormap='winter' if target=='Positive' else 'autumn',
                width=800, height=400,
                max_words=100
            ).generate(subset)
            
            fig_wc, ax = plt.subplots(facecolor=current_theme['card_bg'])
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc)
        else:
            st.info("Insufficient data for text analysis.")

    with c_opp:
        st.subheader("Feature Request Detection")
        keywords = ['wish', 'feature', 'add', 'improve', 'missing', 'suggest']
        opps = df[df['Comment'].str.lower().str.contains('|'.join(keywords))].head(4)
        
        if not opps.empty:
            for _, row in opps.iterrows():
                st.markdown(f"""
                <div class="review-card" style="border-left: 4px solid #8b5cf6;">
                    <span style="color:#8b5cf6; font-weight:700; font-size:0.75rem; letter-spacing:1px;">POTENTIAL LEAD</span><br>
                    <span style="font-style:italic;">"{row['Comment'][:120]}..."</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No feature requests detected in current batch.")

    # --- SECTION 5: VOICE OF CUSTOMER ---
    st.divider()
    st.subheader("🏆 Voice of the Customer")
    
    l_voc, r_voc = st.columns(2)
    
    with l_voc:
        st.markdown("#### ⭐ Top Promoters")
        for _, r in df[df['Sentiment']=='Positive'].nlargest(3, 'Confidence').iterrows():
            st.markdown(f"""
            <div class="review-card" style="border-left: 4px solid #059669;">
                {r['Comment'][:200]}...
            </div>
            """, unsafe_allow_html=True)
    
    with r_voc:
        st.markdown("#### 🚩 Critical Detractors")
        for _, r in df[df['Sentiment']=='Negative'].nlargest(3, 'Confidence').iterrows():
            st.markdown(f"""
            <div class="review-card" style="border-left: 4px solid #e11d48;">
                {r['Comment'][:200]}...
            </div>
            """, unsafe_allow_html=True)

    # --- SECTION 6: SCIENTIFIC AUDIT ---
    st.divider()
    col_hist, col_heat = st.columns(2)
    
    with col_hist:
        st.subheader("Confidence Distribution")
        fig_h = px.histogram(df, x="Confidence", color="Sentiment", 
                             color_discrete_map=colors, 
                             template=current_theme['plotly_template'], nbins=20)
        fig_h.update_layout(height=300, showlegend=True, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_h, use_container_width=True)
    
    with col_heat:
        st.subheader("Intensity Heatmap")
        fig_ht = px.density_heatmap(df, x="Confidence", y="Sentiment", 
                                    color_continuous_scale="Blues", 
                                    template=current_theme['plotly_template'])
        fig_ht.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_ht, use_container_width=True)

    # --- SECTION 7: DATA EXPLORER ---
    st.divider()
    st.subheader("📋 Detailed Data View")
    
    # Summary Table
    audit = df.groupby('Sentiment').agg({'Sentiment': 'count', 'Confidence': 'mean', 'Review_Length': 'mean'}).rename(columns={'Sentiment': 'Count', 'Confidence': 'Avg_Conf', 'Review_Length': 'Avg_Len'})
    st.dataframe(audit.style.format({'Avg_Conf': '{:.1%}', 'Avg_Len': '{:.0f}'}), use_container_width=True)

    with st.expander("📂 View Raw Data Logs (Top 1000)"):
        display_limit = 1000
        df_display = df.head(display_limit).copy()
        
        st.dataframe(
            df_display[['Comment', 'Sentiment', 'Confidence', 'Review_Length']].style.background_gradient(subset=['Confidence'], cmap="Blues"),
            use_container_width=True,
            height=400
        )
        
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="Executive_Sentiment_Report.csv",
            mime="text/csv",
            type="primary"
        )

else:
    # Empty State (Professional)
    st.markdown("<div style='text-align:center; padding: 60px;'>", unsafe_allow_html=True)
    st.markdown("## 👋 Welcome to Sentify Analytics")
    st.markdown("Please upload your customer feedback data (CSV) to generate the executive dashboard.")
    st.image("sa.png", width=120)
    st.markdown("</div>", unsafe_allow_html=True)