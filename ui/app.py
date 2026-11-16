import os
import streamlit as st
import requests
import time
import uuid
import logfire
from dotenv import load_dotenv

# Load environment variables explicitly from the root directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)

# Initialize Logfire without writing Unicode spans to Windows cp1252 consoles.
try:
    token = os.getenv("LOGFIRE_TOKEN")
    export_setting = os.getenv("LOGFIRE_SEND_TO_LOGFIRE", "false").lower()
    if export_setting not in {"true", "false", "if-token-present"}:
        export_setting = "if-token-present"
    send_to_logfire = {"true": True, "false": False}.get(export_setting, "if-token-present")
    logfire.configure(
        token=token,
        send_to_logfire=send_to_logfire,
        console=False,
        service_name="enterprise-rag-ui",
    )
    # logfire.instrument_requests() # Disabled due to OpenTelemetry bug on Windows: MeterProvider.get_meter() got multiple values for argument 'version'
    LOGFIRE_STATUS = "Connected & Tracing" if token and send_to_logfire is not False else "Local only"
except Exception as e:
    print(f"Logfire Init Error in UI: {e}")
    LOGFIRE_STATUS = f"Standby (Error: {e})"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)

# --- AVATARS ---
AI_AVATAR = "🤖"
USER_AVATAR = "👤"

# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"✨ New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Agent OS")
    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")
    auth_mode = os.getenv("AUTH_MODE", "none").strip().lower()
    cognito_token = ""
    if auth_mode == "cognito":
        try:
            is_logged_in = st.user.is_logged_in
        except Exception:
            st.error("Cognito login is not configured. Add the OIDC settings to .streamlit/secrets.toml.")
            st.stop()
        if not is_logged_in:
            st.button(
                "Sign in with Cognito",
                on_click=st.login,
                args=("cognito",),
                use_container_width=True,
            )
            st.stop()
        token_expiry = int(st.user.get("exp", 0))
        if token_expiry <= int(time.time()):
            st.logout()
            st.stop()
        try:
            cognito_token = st.user.tokens["id"]
        except Exception:
            st.error("The Cognito ID token is unavailable; enable expose_tokens = ['id'] in .streamlit/secrets.toml.")
            st.stop()
        st.caption(f"Signed in: {st.user.get('email', 'authenticated user')}")
        st.button("Sign out", on_click=st.logout, use_container_width=True)

    if st.button("🗑️ Clear History & Memory", use_container_width=True, type="primary"):
        auth_token = cognito_token if auth_mode == "cognito" else os.getenv("API_AUTH_TOKEN", "")
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        try:
            response = requests.delete(
                f"{os.getenv('BACKEND_URL', 'http://localhost:8000').rstrip('/')}/memory",
                headers=headers,
                json={"thread_id": st.session_state.session_id},
                timeout=15,
            )
            response.raise_for_status()
        except requests.RequestException:
            st.error("Could not erase server-side conversation memory. Please retry.")
        else:
            logfire.warn("Conversation memory deletion requested")
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

# --- MAIN CHAT ---
st.title("🤖 Enterprise Agentic Assistant")

# Display history
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documentation..."):
    # START TRACE: User Interaction
    # Keep raw user prompts out of exported telemetry. They may contain private
    # document content; request IDs and route-level metrics provide diagnostics.
    with logfire.span("💬 User Chat Interaction", session_id=st.session_state.session_id):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant", avatar=AI_AVATAR):
            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                try:
                    # DISTRIBUTED TRACE: Calling Backend
                    with logfire.span("📡 Calling RAG Backend"):
                        # Get backend URL from env, or default to local if not set
                        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                        url = f"{base_url}/query"
                        payload = {"q": prompt, "thread_id": st.session_state.session_id}
                        auth_token = cognito_token if auth_mode == "cognito" else os.getenv("API_AUTH_TOKEN", "")
                        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
                        response = requests.post(url, json=payload, headers=headers, timeout=60)
                        response.raise_for_status()
                        data = response.json()

                    if not isinstance(data, dict):
                        raise ValueError("The backend returned an invalid response.")
                    full_answer = data.get("answer")
                    if not isinstance(full_answer, str) or not full_answer.strip():
                        raise ValueError("The backend returned an empty answer.")

                    # Show Reasoning Steps from Backend
                    steps = data.get("thought_process", [])
                    for step in steps:
                        st.write(f"⚙️ {step}")

                    status.update(label="✅ Answer Synthesized", state="complete", expanded=False)

                    # --- SHOW SOURCES (NESTED EXPANDABLES) ---
                    sources = data.get("sources", [])
                    if sources:
                        citation_rows = data.get("citations", [])
                        if citation_rows:
                            st.caption(
                                "Cited evidence: "
                                + " · ".join(
                                    row.get("source", f"[{row.get('id')}]")
                                    for row in citation_rows
                                )
                            )
                        with st.expander("📄 View Retrieved Context (Sources)"):
                            for i, source in enumerate(sources):
                                # Create a preview title for each chunk
                                preview = source[:100].replace("\n", " ") + "..."
                                with st.expander(f"Chunk {i+1}: {preview}"):
                                    st.info(source)
                except Exception as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Request Failed", state="error")
                    if isinstance(e, requests.HTTPError) and e.response is not None:
                        detail = {}
                        try:
                            detail = e.response.json().get("detail", {})
                        except (ValueError, AttributeError):
                            pass
                        code = detail.get("code", "backend_error") if isinstance(detail, dict) else "backend_error"
                        reference = detail.get("reference") if isinstance(detail, dict) else None
                        if code == "provider_daily_quota":
                            message = (
                                "The configured LLM provider's daily token quota is exhausted. Resume the saved evaluation "
                                "after quota resets or configure a provider with available quota."
                            )
                        else:
                            message = f"The backend returned HTTP {e.response.status_code} ({code})."
                        if reference:
                            message += f" Reference: {reference}."
                        st.error(message)
                    elif isinstance(e, ValueError) and str(e) == "The backend returned an empty answer.":
                        status.update(label="⚠️ Empty Answer", state="error")
                        st.error(str(e))
                    else:
                        st.error("Could not reach the backend or read its response.")
                    st.stop()

            # Final Answer Streaming
            answer_placeholder = st.empty()
            curr_text = ""
            for char in full_answer:
                curr_text += char
                answer_placeholder.markdown(curr_text + "▌")
                time.sleep(0.005)

            answer_placeholder.markdown(full_answer)
            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            logfire.info("✅ Chat cycle completed successfully.")
