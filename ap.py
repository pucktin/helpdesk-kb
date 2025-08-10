import streamlit as st
from openai import OpenAI
import pinecone
import os

# Initialize API keys from Streamlit Secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment=st.secrets["PINECONE_ENV"])

index = pinecone.Index(st.secrets["PINECONE_INDEX"])

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def search_kb(query, cf_vms_filter=None, top_k=10):
    query_embedding = get_embedding(query)
    filter_dict = {"CF_VMS": {"$eq": cf_vms_filter}} if cf_vms_filter else None

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )
    return results

def summarize_answer(results):
    # Get ticket comments
    comments = [match["metadata"]["comments"] for match in results["matches"]]
    ticket_ids = [match["metadata"]["ticket_id"] for match in results["matches"]]

    # Summarize with GPT
    summary_prompt = f"Summarize the following helpdesk ticket comments into a short answer:\n\n{comments}\n\n"
    summary_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpdesk knowledge base assistant."},
            {"role": "user", "content": summary_prompt}
        ],
        temperature=0.3
    )

    return summary_response.choices[0].message.content, ticket_ids

# Streamlit UI
st.title("Helpdesk Knowledge Base Bot")
st.write("Ask me a question, and I'll find answers from the KB.")

question = st.text_input("Enter your question:")
cf_vms_filter = st.text_input("Filter by tool (e.g., CF_VMS) or leave blank:")
top_k = st.slider("Number of results to search", 3, 20, 10)

if st.button("Search KB"):
    if question.strip():
        st.write("Searching knowledge base, please wait...")
        results = search_kb(question, cf_vms_filter=cf_vms_filter, top_k=top_k)
        summary, ticket_ids = summarize_answer(results)
        
        st.subheader("=== Helpdesk KB Response ===")
        st.write(summary)
        st.write("**Referenced Ticket IDs:**", ", ".join(ticket_ids))
    else:
        st.warning("Please enter a question.")
