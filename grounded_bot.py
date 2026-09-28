# CSC-128 Module 6
# Grounded policy bot
# This bot only answers questions using information retrieved
# from my knowledge base.

import streamlit as st
from groq import Groq

from retriever import Retriever


# Message used when the knowledge base does not have an answer.
REFUSAL = "I don't have that information in my knowledge base."


# Set up the retriever.
retriever = Retriever()


def answer_question(question):
    # First search the knowledge base.
    hits = retriever.search(question)

    # If nothing relevant was found, stop here.
    # The model is NOT called when retrieval is empty.
    if not hits:
        return REFUSAL, []

    # Build the context only from retrieved information.
    context = retriever.build_context(hits)

    # Keep track of the sources that were retrieved.
    sources = []

    for document, score in hits:
        source = document["source"]

        if source not in sources:
            sources.append(source)

    # Get the Groq API key from Streamlit secrets.
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    # This prompt tells the model that it can ONLY use
    # information from the retrieved context.
    prompt = f"""
You are a grounded student help bot.

Answer the question using ONLY the information inside the CONTEXT below.

Do not use outside knowledge.
Do not add information that is not in the context.
Do not guess dates, prices, numbers, policies, or other details.

If the context does not contain enough information to answer the question,
respond exactly with:

{REFUSAL}

CONTEXT:
--------------------
{context}
--------------------

QUESTION:
{question}

Give a short and clear answer based only on the context.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()

    return answer, sources


# -----------------------------
# Streamlit page
# -----------------------------

st.set_page_config(
    page_title="Grounded Policy Bot",
    page_icon="📚"
)

st.title("📚 Grounded Policy Bot")

st.write(
    "Ask me a question about the Central Piedmont library information "
    "in my knowledge base."
)

st.caption(
    "The bot only answers from retrieved information. "
    "If the information is not in the knowledge base, it will say so."
)


question = st.text_input(
    "What would you like to know?"
)


if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching..."):
            answer, sources = answer_question(question)

        st.subheader("Answer")
        st.write(answer)

        # Only display sources when something was retrieved.
        if sources:
            st.subheader("Sources")

            for source in sources:
                st.write(f"- {source}")