import streamlit as st
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets.get("PINECONE_ENV")  # optional, e.g. 'us-west1-gcp'
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone client
if PINECONE_ENV:
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        spec=ServerlessSpec(
            cloud="aws",  # or "gcp"
            region=PINECONE_ENV
        )
    )
else:
    pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX_NAME)

st.title("Helpdesk KB Chatbot")

# Initialize session state variables if they don't exist
if 'response' not in st.session_state:
    st.session_state.response = ""
if 'status' not in st.session_state:
    st.session_state.status = ""
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'cf_vms' not in st.session_state:
    st.session_state.cf_vms = ""

def clear_all():
    st.session_state.question = ""
    st.session_state.cf_vms = ""
    st.session_state.response = ""
    st.session_state.status = ""

def search_kb(question, cf_vms_filter):
    st.session_state.status = "Searching knowledge base, please wait..."
    st.session_state.response = ""
    # Embed the question
    embedding = client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Query Pinecone with optional filter
    if cf_vms_filter.strip():
        filter_query = {"CF_VMS": {"$eq": cf_vms_filter.strip()}}
    else:
        filter_query = None

    results = index.query(
        vector=embedding,
        top_k=10,
        filter=filter_query,
        include_metadata=True
    )

    if not results.matches:
        st.session_state.response = "No relevant tickets found."
        st.session_state.status = "Search complete."
        return

    # Build the summary prompt using the matched tickets metadata
    combined_text = ""
    ticket_ids = []
    for match in results.matches:
        meta = match.metadata
        combined_text += f"- {meta.get('IssueKey', 'Unknown')} : {meta.get('Comments', '')}\n"
        ticket_ids.append(meta.get('IssueKey', 'Unknown'))

    prompt = (
        f"Summarize the following helpdesk ticket comments to answer the question: {question}\n\n"
        f"Comments:\n{combined_text}\n\n"
        f"Provide a concise answer and list referenced ticket IDs at the end."
    )

    # Get GPT summary
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant summarizing helpdesk tickets."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    answer = completion.choices[0].message.content

    # Append referenced tickets
    referenced_ids = ", ".join(ticket_ids)
    st.session_state.response = f"{answer}\n\nReferenced Ticket IDs:\n{referenced_ids}"
    st.session_state.status = "Search complete."

# Layout: use a form to capture Enter or button clicks
with st.form(key="search_form"):
    question_input = st.text_input(
        "Enter your question (or leave blank to exit):",
        value=st.session_state.question,
        key="question_input"
    )
    cf_vms_input = st.text_input(
        "Filter by tool (CF_VMS) or leave blank for all:",
        value=st.session_state.cf_vms,
        key="cf_vms_input"
    )
    col1, col2 = st.columns([1,1])
    with col1:
        search_button = st.form_submit_button("Search")
    with col2:
        clear_button = st.form_submit_button("Clear")

if search_button:
    if question_input.strip():
        st.session_state.question = question_input.strip()
        st.session_state.cf_vms = cf_vms_input.strip()
        search_kb(st.session_state.question, st.session_state.cf_vms)
    else:
        st.warning("Please enter a question to search.")

if clear_button:
    clear_all()

# Show response and status
if st.session_state.status:
    st.info(st.session_state.status)
if st.session_state.response:
    st.markdown("=== Helpdesk KB Response ===")
    st.write(st.session_state.response)
    st.markdown("============================")
