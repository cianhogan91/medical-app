import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# 1. Configuration & Secrets
# -----------------------------

# 1. This looks for a .env file locally. On Render, it does nothing (which is fine).
load_dotenv()

# 2. os.getenv automatically checks Render's Environment Variables dashboard.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CALYPSOAI_TOKEN = os.getenv("CALYPSOAI_TOKEN")

# 3. Double check if variables loaded
if not OPENAI_API_KEY or not CALYPSOAI_TOKEN:
    st.error("Missing API Keys! Ensure they are set in Render's Environment Variables.")
    st.stop()


PROJECT_ID = "019d532d-231c-70b4-a1b6-28f103b1e3ca"
API_URL = "https://us1.calypsoai.app/backend/v1/prompts"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "patient_records"

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

  /* ── Design Tokens ── */
  :root {
    --cvs-red:          #CC0000;
    --cvs-red-deep:     #8B0000;
    --cvs-red-rich:     #AA0000;
    --cvs-red-hover:    #B30000;
    --cvs-red-tint:     #FFF1F1;
    --cvs-red-border:   rgba(204,0,0,0.18);
    --cvs-red-ring:     rgba(204,0,0,0.10);

    --bg-page:          #F7F4F2;
    --bg-surface:       #FFFFFF;
    --bg-muted:         #F2EEEC;
    --bg-inset:         #EAE5E2;

    --text-primary:     #18100E;
    --text-secondary:   #5A4A44;
    --text-muted:       #9A8880;
    --text-on-red:      #FFFFFF;

    --border-light:     #E5DEDA;
    --border-med:       #CEC5C0;

    --shadow-xs:        0 1px 2px rgba(60,20,10,.05);
    --shadow-sm:        0 1px 4px rgba(60,20,10,.07), 0 3px 12px rgba(60,20,10,.06);
    --shadow-md:        0 4px 20px rgba(60,20,10,.10), 0 1px 4px rgba(60,20,10,.06);
    --shadow-red:       0 3px 14px rgba(180,0,0,.20);
    --shadow-red-lg:    0 6px 28px rgba(140,0,0,.26);

    --r-xs: 4px;
    --r-sm: 8px;
    --r-md: 12px;
    --r-lg: 16px;

    --font-ui:      'DM Sans', system-ui, sans-serif;
    --font-display: 'DM Serif Display', Georgia, serif;
  }

  /* ── App Shell ── */
  .stApp {
    background: var(--bg-page) !important;
    font-family: var(--font-ui) !important;
    color: var(--text-primary) !important;
  }
  .block-container {
    max-width: 740px !important;
    padding-top: 0 !important;
    padding-bottom: 7rem !important;
  }
  header[data-testid="stHeader"],
  footer,
  #MainMenu { display: none !important; }

  /* ── Global Text ── */
  p, span, div, li, label {
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
  }
  h1, h2, h3 {
    color: var(--text-primary) !important;
    font-family: var(--font-display) !important;
  }

  /* ══════════════════════════════
     TOP NAV BAR
  ══════════════════════════════ */
  .top-bar {
    position: sticky;
    top: 0;
    z-index: 300;
    background: rgba(255,255,255,0.96);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--border-light);
    margin-bottom: 28px;
    box-shadow: var(--shadow-xs);
  }
  .nav-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 13px 2px;
  }
  .nav-left { display: flex; align-items: center; gap: 11px; }

  /* CVS red wordmark block */
  .cvs-mark {
    background: var(--cvs-red);
    color: #fff !important;
    font-family: var(--font-ui) !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em;
    padding: 5px 10px 4px;
    border-radius: var(--r-xs);
    line-height: 1;
    flex-shrink: 0;
  }
  .nav-title {
    font-size: 13.5px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
    margin: 0 0 2px !important;
    letter-spacing: -0.01em;
    line-height: 1;
  }
  .nav-sub {
    font-size: 10.5px !important;
    color: var(--text-muted) !important;
    margin: 0 !important;
    letter-spacing: 0.02em;
    line-height: 1;
  }
  .nav-right { display: flex; align-items: center; gap: 7px; }

  /* Status pill */
  .status-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--bg-muted);
    border: 1px solid var(--border-light);
    border-radius: 20px;
    padding: 4px 11px;
    font-size: 11px !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-ui) !important;
    font-weight: 500 !important;
  }
  .dot-green {
    width: 6px; height: 6px; border-radius: 50%;
    background: #16A34A;
    box-shadow: 0 0 0 2px rgba(22,163,74,.18);
    flex-shrink: 0;
  }

  /* Governance badge */
  .gov-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--cvs-red-tint);
    border: 1px solid var(--cvs-red-border);
    border-radius: 20px;
    padding: 4px 11px;
    font-size: 11px !important;
    color: var(--cvs-red) !important;
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
  }

  /* ══════════════════════════════
     WELCOME HERO CARD
  ══════════════════════════════ */
  .welcome-card {
    background: linear-gradient(130deg, #CC0000 0%, #8B0000 100%);
    border-radius: var(--r-lg);
    padding: 24px 26px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    gap: 18px;
    box-shadow: var(--shadow-red-lg);
    position: relative;
    overflow: hidden;
  }
  .welcome-card::before {
    content: '';
    position: absolute; top: -50px; right: -50px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(255,255,255,0.055);
  }
  .welcome-card::after {
    content: '';
    position: absolute; bottom: -25px; right: 80px;
    width: 90px; height: 90px;
    border-radius: 50%;
    background: rgba(255,255,255,0.035);
  }
  .avatar-circle {
    width: 50px; height: 50px; border-radius: 50%;
    background: rgba(255,255,255,0.16);
    border: 2px solid rgba(255,255,255,0.32);
    display: flex; align-items: center; justify-content: center;
    color: #fff !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    font-family: var(--font-ui) !important;
    flex-shrink: 0;
  }
  .welcome-name {
    font-family: var(--font-display) !important;
    font-size: 21px !important;
    color: #fff !important;
    margin: 0 0 3px !important;
    line-height: 1.2;
  }
  .welcome-sub {
    font-size: 12.5px !important;
    color: rgba(255,255,255,0.68) !important;
    margin: 0 !important;
    font-weight: 400 !important;
  }

  /* ══════════════════════════════
     PRESCRIPTION METRIC CARDS
  ══════════════════════════════ */
  .rx-section-label {
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
    margin: 0 0 10px !important;
    font-family: var(--font-ui) !important;
  }
  [data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--r-md) !important;
    padding: 16px 18px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: box-shadow 0.18s, border-color 0.18s;
  }
  [data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-md) !important;
    border-color: var(--cvs-red-border) !important;
  }
  [data-testid="stMetricLabel"] p {
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.11em !important;
    color: var(--text-muted) !important;
    font-weight: 700 !important;
  }
  [data-testid="stMetricValue"] {
    font-size: 24px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    font-family: var(--font-display) !important;
    letter-spacing: -0.02em !important;
  }
  [data-testid="stMetricDelta"] {
    background: var(--cvs-red-tint) !important;
    color: var(--cvs-red) !important;
    font-size: 10.5px !important;
    font-weight: 600 !important;
    padding: 2px 8px !important;
    border-radius: 20px !important;
    display: inline-block !important;
  }

  /* ══════════════════════════════
     INFO / ALERT
  ══════════════════════════════ */
  .stAlert {
    background: var(--bg-muted) !important;
    border: 1px solid var(--border-light) !important;
    border-left: 3px solid var(--cvs-red) !important;
    border-radius: var(--r-sm) !important;
    font-size: 12.5px !important;
    box-shadow: none !important;
  }
  .stAlert p { color: var(--text-secondary) !important; }

  /* ══════════════════════════════
     BUTTONS — PRIMARY
  ══════════════════════════════ */
  .stButton > button {
    background: var(--cvs-red) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--r-sm) !important;
    padding: 11px 22px !important;
    font-family: var(--font-ui) !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    width: 100%;
    box-shadow: var(--shadow-red) !important;
    transition: background 0.14s, box-shadow 0.14s, transform 0.1s !important;
    letter-spacing: 0.01em;
  }
  .stButton > button:hover {
    background: var(--cvs-red-hover) !important;
    box-shadow: var(--shadow-red-lg) !important;
    transform: translateY(-1px);
  }
  .stButton > button:active {
    background: var(--cvs-red-deep) !important;
    transform: translateY(0);
  }

  /* BUTTONS — GHOST */
  .ghost-btn .stButton > button {
    background: var(--bg-surface) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-light) !important;
    box-shadow: var(--shadow-xs) !important;
    width: auto !important;
    padding: 7px 15px !important;
    font-size: 12px !important;
    font-weight: 400 !important;
  }
  .ghost-btn .stButton > button:hover {
    background: var(--bg-muted) !important;
    border-color: var(--border-med) !important;
    box-shadow: var(--shadow-sm) !important;
    transform: translateY(-1px);
  }

  /* ══════════════════════════════
     CHAT MESSAGES  ← key focus
  ══════════════════════════════ */

  /* Outer wrapper */
  [data-testid="stChatMessage"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--r-md) !important;
    box-shadow: var(--shadow-sm) !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
    transition: box-shadow 0.15s;
  }
  [data-testid="stChatMessage"]:hover {
    box-shadow: var(--shadow-md) !important;
  }

  /* ALL text inside ANY chat bubble — high-contrast, large, readable */
  [data-testid="stChatMessage"] *,
  [data-testid="stChatMessage"] p,
  [data-testid="stChatMessage"] span,
  [data-testid="stChatMessage"] div,
  [data-testid="stChatMessage"] li,
  [data-testid="stChatMessage"] strong,
  [data-testid="stChatMessage"] em,
  [data-testid="stChatMessage"] code,
  [data-testid="stChatMessage"] blockquote {
    color: #18100E !important;           /* explicit dark, no variable fallback */
    font-family: var(--font-ui) !important;
    font-size: 14.5px !important;
    line-height: 1.7 !important;
  }

  /* Bold text in messages — still very readable */
  [data-testid="stChatMessage"] strong {
    font-weight: 600 !important;
    color: #18100E !important;
  }

  /* Italic / tip line */
  [data-testid="stChatMessage"] em {
    color: var(--text-secondary) !important;
    font-style: italic;
  }

  /* Inline code */
  [data-testid="stChatMessage"] code {
    background: var(--bg-inset) !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-size: 13px !important;
    color: var(--cvs-red-deep) !important;
  }

  /* Avatar icon in chat message rows */
  [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] svg,
  [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] svg {
    color: var(--cvs-red) !important;
    fill: var(--cvs-red) !important;
  }

  /* ══════════════════════════════
     CHAT INPUT BAR
  ══════════════════════════════ */
            [data-testid="stChatInput"] {
    bottom: 50px !important; /* Adjust this number to lift it higher */
    background-color: transparent !important;
    z-index: 1000 !important;
  }
  [data-testid="stChatInput"] > div {
    background: var(--bg-surface) !important;
    border: 1.5px solid var(--border-light) !important;
    border-radius: var(--r-md) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  [data-testid="stChatInput"] > div:focus-within {
    border-color: var(--cvs-red) !important;
    box-shadow: 0 0 0 3px var(--cvs-red-ring), var(--shadow-sm) !important;
  }
  [data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
    font-size: 14px !important;
    caret-color: var(--cvs-red) !important;
  }
  [data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
    font-size: 13.5px !important;
  }
  [data-testid="stChatInput"] button {
    background: var(--cvs-red) !important;
    border-radius: var(--r-xs) !important;
  }
  [data-testid="stChatInput"] button:hover {
    background: var(--cvs-red-hover) !important;
  }

  /* ══════════════════════════════
     MISC
  ══════════════════════════════ */
  hr { border-color: var(--border-light) !important; margin: 18px 0 !important; }
  .stSpinner > div { border-top-color: var(--cvs-red) !important; }
  .stSpinner p { color: var(--text-secondary) !important; font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

# ── Warm watermark layer ──
st.markdown("""
<div style="position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;">
  <svg style="position:absolute;top:-90px;right:-90px;opacity:0.025;"
       width="380" height="380" viewBox="0 0 260 260" xmlns="http://www.w3.org/2000/svg">
    <rect x="87" y="0" width="86" height="260" rx="12" fill="#CC0000"/>
    <rect x="0" y="87" width="260" height="86" rx="12" fill="#CC0000"/>
  </svg>
  <svg style="position:absolute;bottom:16%;left:0;width:100%;opacity:0.035;"
       height="56" viewBox="0 0 900 56" xmlns="http://www.w3.org/2000/svg">
    <polyline points="0,28 175,28 195,28 205,4 215,52 225,4 235,52 245,28 360,28 900,28"
      fill="none" stroke="#CC0000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
  <svg style="position:absolute;bottom:36px;left:22px;opacity:0.04;"
       width="84" height="46" viewBox="0 0 110 58" xmlns="http://www.w3.org/2000/svg">
    <text x="0" y="52" font-family="Georgia,serif" font-size="60" fill="#CC0000">Rx</text>
  </svg>
  <svg style="position:absolute;bottom:72px;right:36px;opacity:0.04;transform:rotate(-20deg);"
       width="86" height="34" viewBox="0 0 110 44" xmlns="http://www.w3.org/2000/svg">
    <rect x="0" y="0" width="110" height="44" rx="22" fill="#CC0000"/>
    <rect x="53" y="0" width="3" height="44" fill="#F7F4F2" opacity="0.6"/>
  </svg>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# 4. ChromaDB Helper
# -----------------------------
@st.cache_resource
def load_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)

# -----------------------------
# 5. OpenAI Embedding + RAG
# -----------------------------
def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """Generate an embedding vector using OpenAI."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def retrieve_context(query: str, top_k: int = 2) -> str:
    """Retrieves patient context from ChromaDB using an OpenAI embedding."""
    try:
        collection = load_chroma_collection()
        q_emb = get_embedding(query)
        results = collection.query(query_embeddings=[q_emb], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        return "\n".join(docs) if docs else "No records found."
    except Exception as e:
        return f"Retrieval Error: {str(e)}"

# -----------------------------
# 6. CalypsoAI Governance Call
# -----------------------------
def calypso_send(text: str) -> dict:
    """Sends the prompt through the CalypsoAI Shield and returns the response."""
    headers = {
        "Authorization": f"Bearer {CALYPSOAI_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"input": text, "project": PROJECT_ID}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, verify=True)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# 7. Session State Init
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ══════════════════════════════
# 8. PAGE 1 — Home Dashboard
# ══════════════════════════════
if st.session_state.page == "home":

    st.markdown("""
    <div class="top-bar">
      <div class="nav-inner">
        <div class="nav-left">
          <span class="cvs-mark">CVS</span>
          <div>
            <p class="nav-title">Secure AI Portal</p>
            <p class="nav-sub">Patient Health Assistant · CalypsoAI Governed</p>
          </div>
        </div>
        <div class="nav-right">
          <span class="status-pill"><span class="dot-green"></span>Systems Active</span>
          <span class="gov-badge">🛡️ Governance On</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-card">
      <div class="avatar-circle">BS</div>
      <div>
        <h2 class="welcome-name">Good morning, Brenda.</h2>
        <p class="welcome-sub">Here's your prescription summary for today</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="rx-section-label">Active Prescriptions</p>', unsafe_allow_html=True)
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
# 9. PAGE 2 — Secure Chat
# ══════════════════════════════
elif st.session_state.page == "chat":

    st.markdown("""
    <div class="top-bar">
      <div class="nav-inner">
        <div class="nav-left">
          <span class="cvs-mark">CVS</span>
          <div>
            <p class="nav-title">Secure Clinical Assistant</p>
            <p class="nav-sub">ChromaDB · OpenAI Embeddings · CalypsoAI Shield</p>
          </div>
        </div>
        <div class="nav-right">
          <span class="status-pill"><span class="dot-green"></span>Ready</span>
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
                    "Ask me about your **prescriptions**, **refill dates**, **dosage instructions**, "
                    "or any other medication questions.\n\n"
                    "*Tip: Press Shift + Enter for a new line.*"
                )
            }
        ]

    # Render messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Bottom-anchored chat input
    if prompt := st.chat_input("Ask about your medications, refills, or dosage…"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Checking governance shield…"):
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
                        client = OpenAI(api_key=OPENAI_API_KEY)
                        llm_res = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a friendly, knowledgeable CVS Pharmacist. "
                                        "Use the patient context below to answer accurately and safely. "
                                        "Be concise and clear. Never speculate about medical diagnoses.\n\n"
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