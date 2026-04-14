import os
import streamlit as st
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv  # <--- New Import

# -----------------------------
# 1. Configuration & Secrets
# -----------------------------
# Load variables from .env file into the environment
load_dotenv() 

# Retrieve keys using os.getenv
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
CALYPSO_TOKEN = os.getenv("CALYPSOAI_TOKEN")

# Verify the keys exist
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
# 3. Global Dark Theme Styles
# -----------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

  /* Base */
  .stApp {
    background-color: #0d1117 !important;
    font-family: 'Inter', sans-serif !important;
    color: #e6edf3 !important;
  }
  .block-container {
    max-width: 780px !important;
    padding-top: 0 !important;
    padding-bottom: 6rem !important;
  }

  /* Hide default Streamlit chrome */
  header[data-testid="stHeader"] { display: none !important; }
  footer { display: none !important; }
  #MainMenu { display: none !important; }

  /* All base text light */
  p, span, div, li, label { color: #e6edf3 !important; font-family: 'Inter', sans-serif !important; }
  h1, h2, h3 { color: #e6edf3 !important; font-family: 'Inter', sans-serif !important; }

  /* Sticky top bar */
  .top-bar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 14px 0 12px;
    margin-bottom: 24px;
  }
  .top-bar-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .top-bar-title {
    font-size: 16px;
    font-weight: 600;
    color: #e6edf3 !important;
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 0 3px;
  }
  .top-bar-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #cc0000;
    display: inline-block;
    flex-shrink: 0;
  }
  .top-bar-sub {
    font-size: 12px !important;
    color: #8b949e !important;
    margin: 0 !important;
  }
  .top-bar-right { display: flex; align-items: center; gap: 8px; }

  /* Status pill */
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 12px !important;
    color: #8b949e !important;
  }
  .status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #3fb950;
  }

  /* Governance badge */
  .gov-label {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 4px 10px;
    font-size: 11px !important;
    color: #cc4444 !important;
  }

  /* Metric / Prescription cards */
  [data-testid="stMetric"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    padding: 16px 18px !important;
  }
  [data-testid="stMetricLabel"] p {
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #8b949e !important;
  }
  [data-testid="stMetricValue"] {
    font-size: 24px !important;
    font-weight: 600 !important;
    color: #e6edf3 !important;
  }
  [data-testid="stMetricDelta"] { color: #cc0000 !important; font-size: 12px !important; }

  /* Info/Alert boxes */
  .stAlert {
    background: #1c2128 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-size: 13px !important;
  }
  .stAlert p { color: #8b949e !important; }

  /* Primary buttons */
  .stButton > button {
    background: #cc0000 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    width: 100%;
    transition: background 0.15s;
  }
  .stButton > button:hover { background: #a80000 !important; }

  /* Secondary / ghost buttons */
  .ghost-btn .stButton > button {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    width: auto !important;
    padding: 6px 14px !important;
    font-size: 12px !important;
  }
  .ghost-btn .stButton > button:hover { background: #30363d !important; }

  /* Chat messages */
  [data-testid="stChatMessage"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    padding: 14px 18px !important;
    color: #e6edf3 !important;
  }
  [data-testid="stChatMessage"] p,
  [data-testid="stChatMessage"] span,
  [data-testid="stChatMessage"] div,
  [data-testid="stChatMessage"] li,
  [data-testid="stChatMessage"] strong,
  [data-testid="stChatMessage"] em {
    color: #e6edf3 !important;
    font-size: 14px !important;
  }

  /* Native chat input — bottom of page */
  [data-testid="stChatInput"] > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
  }
  [data-testid="stChatInput"] textarea {
    background: #161b22 !important;
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    caret-color: #cc0000 !important;
  }
  [data-testid="stChatInput"] textarea::placeholder { color: #8b949e !important; }
  [data-testid="stChatInput"] button { background: #cc0000 !important; border-radius: 6px !important; }

  /* Divider */
  hr { border-color: #30363d !important; }

  /* Spinner */
  .stSpinner > div { border-top-color: #cc0000 !important; }

  /* Home card */
  .home-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .home-card h2 {
    font-size: 19px !important;
    font-weight: 600 !important;
    color: #e6edf3 !important;
    margin: 0 0 3px !important;
  }
  .home-card .sub { font-size: 13px !important; color: #8b949e !important; margin: 0 !important; }
  .avatar-circle {
    width: 44px; height: 44px; border-radius: 50%;
    background: #cc0000;
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-size: 14px; font-weight: 600; flex-shrink: 0;
  }
</style>
""", unsafe_allow_html=True)

# ── Health watermarks (fixed layer behind everything) ──
st.markdown("""
<div style="position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;">

  <!-- ECG / heartbeat line -->
  <svg style="position:absolute;bottom:22%;left:0;width:100%;opacity:0.05;"
       height="80" viewBox="0 0 900 80" xmlns="http://www.w3.org/2000/svg">
    <polyline
      points="0,40 140,40 170,40 183,6 196,74 209,6 222,74 235,40 310,40 900,40"
      fill="none" stroke="#cc0000" stroke-width="2.5"
      stroke-linecap="round" stroke-linejoin="round"/>
  </svg>

  <!-- Doctor silhouette — bottom right -->
  <svg style="position:absolute;bottom:0;right:0;opacity:0.055;"
       width="280" height="430" viewBox="0 0 300 460"
       xmlns="http://www.w3.org/2000/svg" fill="#cc0000">
    <ellipse cx="150" cy="62" rx="40" ry="46"/>
    <rect x="130" y="102" width="40" height="26" rx="7"/>
    <path d="M50 140 Q70 130 130 130 Q138 130 143 138 L150 158 L157 138 Q162 130 170 130 L230 130 Q265 140 248 178 L240 290 Q238 308 220 308 L80 308 Q62 308 60 290 Z"/>
    <path d="M130 130 L105 178 L138 165 Z"/>
    <path d="M170 130 L195 178 L162 165 Z"/>
    <path d="M116 158 Q88 190 86 226 Q84 250 101 256 Q122 264 126 242 Q128 228 116 226 Q107 224 108 235"
      stroke="#cc0000" stroke-width="6" fill="none" stroke-linecap="round"/>
    <circle cx="108" cy="240" r="9"/>
    <path d="M230 142 Q266 160 272 212 Q276 240 260 245 L242 248 Q228 248 224 232 L218 196 Z"/>
    <path d="M70 142 Q34 160 28 212 Q24 240 40 245 L58 248 Q72 248 76 232 L82 196 Z"/>
    <rect x="14" y="238" width="46" height="58" rx="5"/>
    <rect x="20" y="228" width="34" height="10" rx="4"/>
    <rect x="22" y="250" width="30" height="3" rx="1"/>
    <rect x="22" y="260" width="22" height="3" rx="1"/>
    <rect x="22" y="270" width="26" height="3" rx="1"/>
    <rect x="98" y="306" width="46" height="106" rx="9"/>
    <rect x="156" y="306" width="46" height="106" rx="9"/>
    <ellipse cx="121" cy="412" rx="30" ry="12"/>
    <ellipse cx="179" cy="412" rx="30" ry="12"/>
    <rect x="165" y="195" width="38" height="28" rx="5"/>
    <rect x="178" y="190" width="5" height="18" rx="2"/>
    <rect x="188" y="190" width="5" height="14" rx="2"/>
  </svg>

  <!-- Pharmacy cross — top left -->
  <svg style="position:absolute;top:-50px;left:-50px;opacity:0.04;"
       width="260" height="260" viewBox="0 0 260 260" xmlns="http://www.w3.org/2000/svg">
    <rect x="87" y="0" width="86" height="260" rx="10" fill="#cc0000"/>
    <rect x="0" y="87" width="260" height="86" rx="10" fill="#cc0000"/>
  </svg>

  <!-- Pill shape — top right -->
  <svg style="position:absolute;top:70px;right:50px;opacity:0.05;transform:rotate(-28deg);"
       width="110" height="44" viewBox="0 0 110 44" xmlns="http://www.w3.org/2000/svg">
    <rect x="0" y="0" width="110" height="44" rx="22" fill="#cc0000"/>
    <rect x="53" y="0" width="4" height="44" fill="#0d1117" opacity="0.4"/>
  </svg>

  <!-- Rx — bottom left -->
  <svg style="position:absolute;bottom:50px;left:30px;opacity:0.05;"
       width="110" height="58" viewBox="0 0 110 58" xmlns="http://www.w3.org/2000/svg">
    <text x="0" y="52" font-family="Georgia,serif" font-size="60" fill="#cc0000">Rx</text>
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
        <div>
          <div class="top-bar-title"><span class="top-bar-dot"></span>CVS Secure AI Portal</div>
          <p class="top-bar-sub">ChromaDB + CalypsoAI &bull; Patient Health Assistant</p>
        </div>
        <div class="top-bar-right">
          <span class="status-pill"><span class="status-dot"></span>Active</span>
          <span class="gov-label">🛡️ Governance On</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="home-card">
      <div class="avatar-circle">BS</div>
      <div>
        <h2>Good morning, Brenda.</h2>
        <p class="sub">Here's your prescription summary for today</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Metformin", "60 tabs", "Due May 15")
    with col2:
        st.metric("Lisinopril", "30 tabs", "Due Apr 20")

    st.divider()
    st.info("🔒 Your data is protected by CVS Secure AI Governance.")

    if st.button("💬 Open Secure Clinical Chat"):
        st.session_state.page = "chat"
        st.rerun()

# ══════════════════════════════
# 8. PAGE 2 — Secure Chat
# ══════════════════════════════
elif st.session_state.page == "chat":

    st.markdown("""
    <div class="top-bar">
      <div class="top-bar-inner">
        <div>
          <div class="top-bar-title"><span class="top-bar-dot"></span>Secure Clinical Assistant</div>
          <p class="top-bar-sub">ChromaDB + CalypsoAI &bull; Focused on your health records</p>
        </div>
        <div class="top-bar-right">
          <span class="status-pill"><span class="status-dot"></span>Idle</span>
          <span class="gov-label">🛡️ Governance On</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_back, col_clear = st.columns([1, 1])
    with col_back:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("⬅ Home"):
            st.session_state.page = "home"
            st.session_state.messages = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col_clear:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("🗑 Clear chat"):
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
    if prompt := st.chat_input("Type your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Checking safety protocols..."):
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