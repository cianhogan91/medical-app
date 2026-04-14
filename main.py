import os
import streamlit as st
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# -----------------------------
# 1. Configuration & Secrets
# -----------------------------
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
CALYPSO_TOKEN = os.getenv("CALYPSOAI_TOKEN")

if not OPENAI_KEY or not CALYPSO_TOKEN:
    st.error("Missing Secrets! Ensure OPENAI_API_KEY and CALYPSOAI_TOKEN are set in your .env file.")
    st.stop()

PROJECT_ID = "019d532d-231c-70b4-a1b6-28f103b1e3ca"
API_URL = "https://us1.calypsoai.app/backend/v1/prompts"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "patient_records"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SECURITY_BLOCK_MSG = "🚨 **SECURITY ALERT:** This request was blocked by CVS Pharmacy Governance for a safety violation."

# -----------------------------
# 2. Page Config
# -----------------------------
st.set_page_config(page_title="CVS Secure AI Portal", layout="centered")

# -----------------------------
# 3. Professional CVS Theme
# -----------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Serif+Display:ital@0;1&display=swap');

  /* ── CSS Tokens ── */
  :root {
    --cvs-red:        #CC0000;
    --cvs-red-deep:   #9B0000;
    --cvs-red-rich:   #B30000;
    --cvs-red-soft:   #E8000010;
    --cvs-red-border: #CC000028;

    --bg-base:        #FAFAF9;
    --bg-surface:     #FFFFFF;
    --bg-muted:       #F5F2F0;
    --bg-inset:       #EDE9E6;

    --text-primary:   #1A1210;
    --text-secondary: #5C4F4A;
    --text-muted:     #9C8D87;
    --text-on-red:    #FFFFFF;

    --border-base:    #E2DBD8;
    --border-strong:  #C9BCB8;

    --shadow-card:    0 1px 3px rgba(80,30,20,.06), 0 4px 16px rgba(80,30,20,.08);
    --shadow-lift:    0 4px 24px rgba(80,30,20,.12);

    --radius-sm:  6px;
    --radius-md:  10px;
    --radius-lg:  16px;
    --radius-xl:  24px;

    --font-display: 'DM Serif Display', Georgia, serif;
    --font-body:    'DM Sans', system-ui, sans-serif;
  }

  /* ── Reset & Base ── */
  .stApp {
    background: var(--bg-base) !important;
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
  }
  .block-container {
    max-width: 760px !important;
    padding-top: 0 !important;
    padding-bottom: 7rem !important;
  }

  /* Hide Streamlit chrome */
  header[data-testid="stHeader"],
  footer,
  #MainMenu { display: none !important; }

  /* Universal text reset */
  p, span, div, li, label {
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
  }
  h1, h2, h3 {
    color: var(--text-primary) !important;
    font-family: var(--font-display) !important;
  }

  /* ── Top Navigation Bar ── */
  .top-bar {
    position: sticky;
    top: 0;
    z-index: 200;
    background: rgba(255,255,255,0.94);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border-base);
    padding: 0;
    margin-bottom: 32px;
  }
  .top-bar-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 4px;
  }
  .top-bar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .cvs-wordmark {
    background: var(--cvs-red);
    color: #fff !important;
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em;
    padding: 4px 10px;
    border-radius: 4px;
    line-height: 1;
  }
  .top-bar-title {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    margin: 0 !important;
    letter-spacing: -0.01em;
  }
  .top-bar-sub {
    font-size: 11px !important;
    color: var(--text-muted) !important;
    margin: 2px 0 0 !important;
    letter-spacing: 0.02em;
  }
  .top-bar-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* ── Pills & Badges ── */
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-muted);
    border: 1px solid var(--border-base);
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 11.5px !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
  }
  .status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #22C55E;
    box-shadow: 0 0 0 2px #22C55E30;
    flex-shrink: 0;
  }
  .gov-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #FFF1F1;
    border: 1px solid var(--cvs-red-border);
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 11.5px !important;
    color: var(--cvs-red) !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em;
  }

  /* ── Hero Welcome Card ── */
  .welcome-card {
    background: linear-gradient(135deg, var(--cvs-red) 0%, var(--cvs-red-deep) 100%);
    border-radius: var(--radius-lg);
    padding: 28px 28px 28px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 6px 32px rgba(180,0,0,.22), 0 1px 4px rgba(180,0,0,.14);
    position: relative;
    overflow: hidden;
  }
  .welcome-card::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
  }
  .welcome-card::after {
    content: '';
    position: absolute;
    bottom: -30px; right: 60px;
    width: 100px; height: 100px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
  }
  .welcome-avatar {
    width: 52px; height: 52px;
    border-radius: 50%;
    background: rgba(255,255,255,0.18);
    border: 2px solid rgba(255,255,255,0.35);
    display: flex; align-items: center; justify-content: center;
    color: #fff !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    font-family: var(--font-body) !important;
    flex-shrink: 0;
    backdrop-filter: blur(4px);
  }
  .welcome-name {
    font-family: var(--font-display) !important;
    font-size: 22px !important;
    color: #fff !important;
    margin: 0 0 4px !important;
    line-height: 1.2;
  }
  .welcome-sub {
    font-size: 13px !important;
    color: rgba(255,255,255,0.72) !important;
    margin: 0 !important;
    font-weight: 400 !important;
  }

  /* ── Metric Cards ── */
  [data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-base) !important;
    border-radius: var(--radius-md) !important;
    padding: 18px 20px !important;
    box-shadow: var(--shadow-card) !important;
    transition: box-shadow 0.2s, border-color 0.2s;
  }
  [data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-lift) !important;
    border-color: var(--cvs-red-border) !important;
  }
  [data-testid="stMetricLabel"] p {
    font-size: 10.5px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-family: var(--font-body) !important;
  }
  [data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    font-family: var(--font-display) !important;
    letter-spacing: -0.02em !important;
  }
  [data-testid="stMetricDelta"] {
    color: var(--cvs-red) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    font-family: var(--font-body) !important;
    background: #FFF1F1;
    padding: 2px 7px;
    border-radius: 20px;
    display: inline-block;
  }

  /* ── Info / Alert boxes ── */
  .stAlert {
    background: var(--bg-muted) !important;
    border: 1px solid var(--border-base) !important;
    border-left: 3px solid var(--cvs-red) !important;
    border-radius: var(--radius-md) !important;
    font-size: 13px !important;
    box-shadow: none !important;
  }
  .stAlert p {
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
  }

  /* ── Buttons — Primary ── */
  .stButton > button {
    background: var(--cvs-red) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 11px 22px !important;
    font-family: var(--font-body) !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em;
    width: 100%;
    box-shadow: 0 2px 8px rgba(180,0,0,.20) !important;
    transition: background 0.15s, box-shadow 0.15s, transform 0.1s !important;
  }
  .stButton > button:hover {
    background: var(--cvs-red-rich) !important;
    box-shadow: 0 4px 16px rgba(180,0,0,.28) !important;
    transform: translateY(-1px);
  }
  .stButton > button:active {
    background: var(--cvs-red-deep) !important;
    transform: translateY(0);
  }

  /* ── Buttons — Ghost ── */
  .ghost-btn .stButton > button {
    background: var(--bg-surface) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-base) !important;
    box-shadow: var(--shadow-card) !important;
    width: auto !important;
    padding: 7px 16px !important;
    font-size: 12.5px !important;
    font-weight: 400 !important;
  }
  .ghost-btn .stButton > button:hover {
    background: var(--bg-muted) !important;
    border-color: var(--border-strong) !important;
    box-shadow: var(--shadow-lift) !important;
    transform: translateY(-1px);
  }

  /* ── Chat Messages ── */
  [data-testid="stChatMessage"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-base) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-card) !important;
    padding: 16px 20px !important;
    margin-bottom: 8px !important;
    transition: box-shadow 0.15s;
  }
  [data-testid="stChatMessage"]:hover {
    box-shadow: var(--shadow-lift) !important;
  }
  [data-testid="stChatMessage"] p,
  [data-testid="stChatMessage"] span,
  [data-testid="stChatMessage"] div,
  [data-testid="stChatMessage"] li,
  [data-testid="stChatMessage"] strong,
  [data-testid="stChatMessage"] em {
    color: var(--text-primary) !important;
    font-size: 14px !important;
    font-family: var(--font-body) !important;
    line-height: 1.65 !important;
  }

  /* ── Chat Input ── */
  [data-testid="stChatInput"] > div {
    background: var(--bg-surface) !important;
    border: 1.5px solid var(--border-base) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-card) !important;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  [data-testid="stChatInput"] > div:focus-within {
    border-color: var(--cvs-red) !important;
    box-shadow: 0 0 0 3px rgba(204,0,0,0.08), var(--shadow-card) !important;
  }
  [data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 14px !important;
    caret-color: var(--cvs-red) !important;
  }
  [data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
  }
  [data-testid="stChatInput"] button {
    background: var(--cvs-red) !important;
    border-radius: var(--radius-sm) !important;
    transition: background 0.15s !important;
  }
  [data-testid="stChatInput"] button:hover {
    background: var(--cvs-red-rich) !important;
  }

  /* ── Divider ── */
  hr { border-color: var(--border-base) !important; margin: 20px 0 !important; }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: var(--cvs-red) !important; }

  /* ── Section Label ── */
  .section-label {
    font-size: 10.5px !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
    margin: 0 0 12px !important;
    font-family: var(--font-body) !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Subtle warm-toned watermark layer ──
st.markdown("""
<div style="position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;">

  <!-- Pharmacy cross — large, top right, very subtle -->
  <svg style="position:absolute;top:-80px;right:-80px;opacity:0.03;"
       width="340" height="340" viewBox="0 0 260 260" xmlns="http://www.w3.org/2000/svg">
    <rect x="87" y="0" width="86" height="260" rx="12" fill="#CC0000"/>
    <rect x="0" y="87" width="260" height="86" rx="12" fill="#CC0000"/>
  </svg>

  <!-- ECG line — bottom strip -->
  <svg style="position:absolute;bottom:18%;left:0;width:100%;opacity:0.04;"
       height="60" viewBox="0 0 900 60" xmlns="http://www.w3.org/2000/svg">
    <polyline
      points="0,30 180,30 200,30 210,5 220,55 230,5 240,55 250,30 370,30 900,30"
      fill="none" stroke="#CC0000" stroke-width="2"
      stroke-linecap="round" stroke-linejoin="round"/>
  </svg>

  <!-- Rx mark — bottom left -->
  <svg style="position:absolute;bottom:40px;left:24px;opacity:0.04;"
       width="90" height="48" viewBox="0 0 110 58" xmlns="http://www.w3.org/2000/svg">
    <text x="0" y="52" font-family="Georgia,serif" font-size="60" fill="#CC0000">Rx</text>
  </svg>

  <!-- Pill — lower right -->
  <svg style="position:absolute;bottom:80px;right:40px;opacity:0.04;transform:rotate(-22deg);"
       width="90" height="36" viewBox="0 0 110 44" xmlns="http://www.w3.org/2000/svg">
    <rect x="0" y="0" width="110" height="44" rx="22" fill="#CC0000"/>
    <rect x="53" y="0" width="4" height="44" fill="#FAFAF9" opacity="0.6"/>
  </svg>

</div>
""", unsafe_allow_html=True)

# -----------------------------
# 4. RAG & Embedding Helpers
# -----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def load_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)

def retrieve_context(query: str, top_k: int = 2):
    """Retrieves patient context from ChromaDB using vector search."""
    try:
        collection = load_chroma_collection()
        model = load_embedder()
        q_emb = model.encode([query])[0].tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        return "\n".join(docs) if docs else "No records found."
    except Exception as e:
        return f"Retrieval Error: {str(e)}"

# -----------------------------
# 5. CalypsoAI Governance Call
# -----------------------------
def calypso_send(text: str):
    """Sends prompt through the CalypsoAI Shield."""
    headers = {
        "Authorization": f"Bearer {CALYPSO_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"input": text, "project": PROJECT_ID}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, verify=True)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# 6. Session State Init
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ══════════════════════════════
# 7. PAGE 1 — Home Dashboard
# ══════════════════════════════
if st.session_state.page == "home":

    st.markdown("""
    <div class="top-bar">
      <div class="top-bar-inner">
        <div class="top-bar-logo">
          <span class="cvs-wordmark">CVS</span>
          <div>
            <p class="top-bar-title">Secure AI Portal</p>
            <p class="top-bar-sub">Patient Health Assistant · CalypsoAI Governed</p>
          </div>
        </div>
        <div class="top-bar-right">
          <span class="status-pill"><span class="status-dot"></span>Systems Active</span>
          <span class="gov-badge">🛡️ Governance On</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-card">
      <div class="welcome-avatar">BS</div>
      <div>
        <h2 class="welcome-name">Good morning, Brenda.</h2>
        <p class="welcome-sub">Here's your prescription summary for today</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">Active Prescriptions</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Metformin", "60 tabs", "Due May 15")
    with col2:
        st.metric("Lisinopril", "30 tabs", "Due Apr 20")

    st.divider()
    st.info("🔒 Your health data is end-to-end protected by CVS Secure AI Governance. All queries are monitored and filtered in real time.")

    if st.button("💬  Open Secure Clinical Chat"):
        st.session_state.page = "chat"
        st.rerun()

# ══════════════════════════════
# 8. PAGE 2 — Secure Chat
# ══════════════════════════════
elif st.session_state.page == "chat":

    st.markdown("""
    <div class="top-bar">
      <div class="top-bar-inner">
        <div class="top-bar-logo">
          <span class="cvs-wordmark">CVS</span>
          <div>
            <p class="top-bar-title">Secure Clinical Assistant</p>
            <p class="top-bar-sub">ChromaDB + CalypsoAI · Focused on your health records</p>
          </div>
        </div>
        <div class="top-bar-right">
          <span class="status-pill"><span class="status-dot"></span>Ready</span>
          <span class="gov-badge">🛡️ Governance On</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_back, col_clear = st.columns([1, 1])
    with col_back:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.session_state.messages = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col_clear:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("🗑  Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Seed opening greeting
    if "messages" not in st.session_state or len(st.session_state.messages) == 0:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hello, Brenda! How can I help you today?\n\n"
                    "Ask me about your prescriptions, refill dates, dosage instructions, "
                    "or any medication questions.\n\n"
                    "*Tip: Shift+Enter for a new line.*"
                )
            }
        ]

    # Render messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Native bottom-anchored chat input
    if prompt := st.chat_input("Ask about your medications, refills, or dosage…"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Checking safety protocols…"):
            response_data = calypso_send(prompt)

            if response_data:
                outcome = str(
                    response_data.get("outcome") or
                    response_data.get("result", {}).get("outcome", "")
                ).lower()

                if "blocked" in outcome:
                    reason = (
                        response_data.get("reason") or
                        response_data.get("result", {}).get("reason", "Policy Violation")
                    )
                    final_msg = f"{SECURITY_BLOCK_MSG}\n\n*Governance Reason: {reason}*"

                else:
                    context = retrieve_context(prompt)
                    try:
                        client = OpenAI(api_key=OPENAI_KEY)
                        llm_res = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a friendly and knowledgeable CVS Pharmacist. "
                                        "Use the patient context below to answer accurately and safely. "
                                        "Be concise and clear.\n\n"
                                        f"Patient Context:\n{context}"
                                    )
                                },
                                {"role": "user", "content": prompt}
                            ]
                        )
                        final_msg = llm_res.choices[0].message.content
                    except Exception as e:
                        final_msg = f"⚠️ LLM Error: {str(e)}"

                st.session_state.messages.append({"role": "assistant", "content": final_msg})
                st.rerun()

            else:
                st.error("Unable to reach Governance Shield. Connection interrupted.")