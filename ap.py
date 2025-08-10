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

st.title("Helpdesk KB Chatbot")

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def ask_question(question, cf_vms_filter=None, top_k=5):
    query_embedding = get_embedding(question)

    # Build filter if cf_vms_filter is specified
    filter_metadata = {}
    if cf_vms_filter:
        filter_metadata = {"CF_VMS": cf_vms_filter}

    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_metadata or None
    )

    # Compose response summary
    answers = []
    ticket_ids = []
    for match in results.matches:
        meta = match.metadata or {}
        text = meta.get("Comments", "")  # or whatever field has content
        ticket_id = meta.get("IssueKey") or meta.get("id") or "Unknown"
        ticket_ids.append(ticket_id)
        answers.append(text)

    summary = (
        f"=== Helpdesk KB Response ===\n"
        f"{' '.join(answers)}\n"
        f"\nReferenced Ticket IDs:\n{', '.join(ticket_ids)}\n"
        f"============================"
    )

    return summary

def main():
    question = st.text_input("Enter your question:")
    cf_vms = st.text_input("Filter by tool (CF_VMS) or leave blank for all:")

    if st.button("Ask") and question.strip():
        with st.spinner("Searching knowledge base, please wait..."):
            answer = ask_question(question, cf_vms_filter=cf_vms.strip() or None)
            st.text_area("Answer:", value=answer, height=300)

if __name__ == "__main__":
    main()
