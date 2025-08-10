import streamlit as st
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets.get("PINECONE_ENV")  # optional, like 'us-west1-gcp'
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)

if PINECONE_ENV:
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_ENV
        )
    )
else:
    pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX_NAME)

st.title("Helpdesk KB Chatbot")

# Initialize session state vars
if "question" not in st.session_state:
    st.session_state.question = ""
if "cf_vms" not in st.session_state:
    st.session_state.cf_vms = ""
if "status" not in st.session_state:
    st.session_state.status = ""
if "response" not in st.session_state:
    st.session_state.response = ""

def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def ask_question(question: str, cf_vms_filter: str = None):
    query_embedding = get_embedding(question)

    filter_dict = None
    if cf_vms_filter and cf_vms_filter.strip():
        filter_dict = {"CF_VMS": {"$eq": cf_vms_filter.strip()}}

    query_response = index.query(
        vector=query_embedding,
        top_k=10,
        filter=filter_dict,
        include_metadata=True
    )

    matches = query_response.matches
    if not matches:
        return "No relevant tickets found in knowledge base."

    combined_text = ""
    ticket_ids = []
    for match in matches:
        metadata = match.metadata or {}
        ticket_ids.append(metadata.get("IssueKey") or metadata.get("id") or "UnknownID")
        combined_text += metadata.get("Comments", "") + "\n"

    prompt = (
        "You are a helpdesk knowledge base assistant. "
        "Given the following comments from past tickets, provide a concise summary answer "
        "to the user's question. Then list the ticket IDs referenced.\n\n"
        f"User question: {question}\n\n"
        f"Ticket comments:\n{combined_text}\n\n"
        "Summary answer:"
    )

    chat_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful and concise helpdesk assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.3,
    )
    summary = chat_response.choices[0].message.content.strip()

    tickets_str = ", ".join(ticket_ids)
    summary += f"\n\nReferenced Ticket IDs:\n{tickets_str}\n\n============================"

    return summary

def clear_search():
    st.session_state.question = ""
    st.session_state.cf_vms = ""
    st.session_state.status = ""
    st.session_state.response = ""

def main():
    with st.form("search_form", clear_on_submit=False):
        question_input = st.text_input("Enter your question (or leave blank to exit):", value=st.session_state.question)
        cf_vms_input = st.text_input("Filter by tool (CF_VMS) or leave blank for all:", value=st.session_state.cf_vms)
        
        submit = st.form_submit_button("Search")
        clear = st.form_submit_button("Clear / Search Again")

        if submit:
            if question_input.strip() == "":
                st.warning("Please enter a question to search.")
            else:
                st.session_state.question = question_input
                st.session_state.cf_vms = cf_vms_input
                st.session_state.status = "Searching knowledge base, please wait..."
                st.experimental_rerun()  # rerun to show status update before running query

        if clear:
            clear_search()
            st.experimental_rerun()

    # Run search if status is searching
    if st.session_state.status == "Searching knowledge base, please wait...":
        # Run query outside form to avoid rerun loop
        st.session_state.response = ask_question(st.session_state.question, st.session_state.cf_vms)
        st.session_state.status = "Search complete."
        st.experimental_rerun()

    if st.session_state.status:
        st.write(st.session_state.status)

    if st.session_state.response:
        st.text_area("Response:", value=st.session_state.response, height=350)

if __name__ == "__main__":
    main()
