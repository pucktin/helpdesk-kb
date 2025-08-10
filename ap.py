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

# Initialize Pinecone client with environment spec if provided
if PINECONE_ENV:
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        spec=ServerlessSpec(
            cloud="aws",  # or "gcp" if your env is on GCP
            region=PINECONE_ENV
        )
    )
else:
    pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to Pinecone index
index = pc.Index(PINECONE_INDEX_NAME)

st.title("Helpdesk KB Chatbot")

def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def search_pinecone(query_embedding, cf_vms_filter=None, top_k=5):
    # Build metadata filter if CF_VMS filter is given
    metadata_filter = None
    if cf_vms_filter and cf_vms_filter.strip():
        metadata_filter = {
            "CF_VMS": {"$eq": cf_vms_filter.strip()}
        }

    query_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter
    )
    return query_response.matches

def generate_summary(question, tickets_text):
    prompt = f"""You are a helpful assistant.

User question: {question}

Here are relevant past tickets for context:
{tickets_text}

Please provide a concise summary answer to the user's question based on the tickets above."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        temperature=0.3,
    )
    return response.choices[0].message.content

def main():
    question = st.text_input("Enter your question (or leave blank to exit):")
    cf_vms = st.text_input("Filter by tool (CF_VMS) or leave blank for all:")

    if question:
        st.info("Searching knowledge base, please wait...")
        query_embedding = get_embedding(question)

        matches = search_pinecone(query_embedding, cf_vms_filter=cf_vms, top_k=10)

        if not matches:
            st.warning("No matching tickets found.")
            return

        # Concatenate the relevant ticket text fields for GPT context
        tickets_text = ""
        ticket_ids = []
        for match in matches:
            meta = match.metadata or {}
            ticket_id = meta.get("IssueKey") or meta.get("id") or "Unknown ID"
            ticket_ids.append(ticket_id)

            # Construct a text snippet per ticket
            snippet = f"Ticket ID: {ticket_id}\nTitle: {meta.get('Title', '')}\nDescription: {meta.get('Description', '')}\nComments: {meta.get('Comments', '')}\n\n"
            tickets_text += snippet

        # Generate GPT summary answer
        answer = generate_summary(question, tickets_text)

        # Display answer and referenced tickets
        st.markdown("=== **Helpdesk KB Response** ===")
        st.write(answer.strip())
        st.markdown("**Referenced Ticket IDs:**")
        st.write(", ".join(ticket_ids))
        st.markdown("============================")

if __name__ == "__main__":
    main()
