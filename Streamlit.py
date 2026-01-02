import streamlit as st
import requests


FASTAPI_URL = "http://127.0.0.1:8000"  # change if running remotely

st.set_page_config(page_title="RAG Chat - Your File Assistant", layout="centered")

st.title("RAG Chat Agent")
st.caption("Upload a document and ask any question based on it.")


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False


# ==========================
# FILE UPLOAD SECTION
# ==========================
st.subheader("Step 1: Upload documents")

uploaded_files = st.file_uploader(
    "Upload PDF, TXT, or MD files (you can select multiple)", 
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

if uploaded_files and not st.session_state.file_uploaded:
    with st.spinner(f"📂 Uploading and processing {len(uploaded_files)} file(s)..."):
        # Prepare multiple files for upload
        files = [("files", (file.name, file, file.type)) for file in uploaded_files]
        
        response = requests.post(f"{FASTAPI_URL}/Upload_File", files=files)

        if response.status_code == 200:
            result = response.json()
            file_list = ", ".join(result.get("filenames", []))
            st.success(f"✅ {result.get('count', 0)} file(s) processed successfully: {file_list}")
            st.session_state.file_uploaded = True
            st.session_state.messages = []  # Clear previous chat history
        else:
            st.error("❌ Failed to process files.")
            st.stop()


# ==========================
# CHAT SECTION
# ==========================
st.subheader("Step 2: Ask a Question")

if not st.session_state.file_uploaded:
    st.warning("⚠️ Please upload a document first before asking questions.")
else:
    # Display previous messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # Chat input
    user_input = st.chat_input("💬 Enter your question:")

    if user_input:
        # Display user message
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get response from FastAPI
        with st.spinner("🤔 Thinking..."):
            res = requests.post(f"{FASTAPI_URL}/chat", json={"question": user_input})

            if res.status_code == 200:
                data = res.json()
                assistant_reply = data.get("Assistant", "...")
            else:
                assistant_reply = f"❌ Error from server: {res.status_code}"

        # Display assistant message
        st.chat_message("assistant").write(assistant_reply)
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        st.rerun()
