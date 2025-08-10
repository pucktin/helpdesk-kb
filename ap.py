import streamlit as st
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets.get("PINECONE_ENV")  # optional, like 'us-west1-gcp'
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone client with environment spec if available
if PINECONE_ENV:
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        spec=ServerlessSpec(
            cloud="aws",   # or "gcp" depending on your setup
            region=PINECONE_ENV
        )
    )
else:
    pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to Pinecone index
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize session state variables on first run
if "question" not in st.session_state:
    st.session_state.question = ""
if "cf_vms" not in st.session_state:
    st.session_state.cf_vms = ""
if "response" not in st.session_state:
    st.session_state.response = ""
if "status" not in st.session_state:
    st.session_state.status = ""
if "run_search" not in st.session_state:
    st.session_state.run_search = False

def clear_search():
    st.session_state.question = ""
    st.session_state.cf_vms = ""
    st.session_state.response = ""
    st.session_state.status = ""
    st.session_state.run_search = False

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def ask_question(question, cf_vms_filter=None):
    if not question.strip():
        return "Please enter a question."

    embedding = get_embedding(question)
    filter_metadata = {"CF_VMS": cf_vms_filter} if cf_vms_filter else None

    query_response = index.query(
        vector=embedding,
        top_k=10,
        include_metadata=True,
        filter=filter_metadata
    )

    matches = query_response.matches
    if not matches:
        return "No relevant tickets found. Please try refining your question or filter."

    combined_text = ""
    ticket_ids = []
    for match in matches:
        md = match.metadata or {}
        combined_text += md.get("Comments", "") + "\n\n"
        ticket_id = md.get("IssueKey") or md.get("id")
        if ticket_id and ticket_id not in ticket_ids:
            ticket_ids.append(ticket_id)

    prompt = (
        "You are a helpful support assistant. Summarize the following relevant ticket information briefly and clearly:\n\n"
        f"{combined_text}\n\n"
        "Provide a clear summary that addresses the user's question. "
        "After the summary, list the referenced ticket IDs.\n"
    )

    chat_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful and concise assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    summary = chat_response.choices[0].message.content.strip()
    tickets_line = "Referenced Ticket IDs:\n" + ", ".join(ticket_ids)

    return f"=== Helpdesk KB Response ===\n\n{summary}\n\n{tickets_line}\n\n============================"

def main():
    st.title("Helpdesk KB Chatbot")

    with st.form("search_form", clear_on_submit=False):
        question_input = st.text_input("Enter your question (or leave blank to exit):", value=st.session_state.question)
        cf_vms_input = st.text_input("Filter by tool (CF_VMS) or leave blank for all:", value=st.session_state.cf_vms)

        search_clicked = st.form_submit_button("Search")
        clear_clicked = st.form_submit_button("Clear")

        if search_clicked:
            if question_input.strip() == "":
                st.warning("Please enter a question to search.")
            else:
                st.session_state.question = question_input
                st.session_state.cf_vms = cf_vms_input
                st.session_state.status = "Searching knowledge base, please wait..."
                st.session_state.run_search = True
                st.experimental_rerun()

        if clear_clicked:
            clear_search()
            # no rerun here, just clear UI

    # Controlled rerun steps for search
    if st.session_state.run_search:
        # Perform the query only once per run_search True
        st.session_state.response = ask_question(st.session_state.question, st.session_state.cf_vms)
        st.session_state.status = "Search complete."
        st.session_state.run_search = False
        st.experimental_rerun()

    # Show status and results
    if st.session_state.status:
        st.write(st.session_state.status)

    if st.session_state.response:
        st.text_area("Response:", value=st.session_state.response, height=350)

if __name__ == "__main__":
    main()
