
import streamlit as st
from rag_pipeline import create_rag_components, ask_question

st.set_page_config(
    page_title="AI & ML Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI & Machine Learning Knowledge Assistant")
st.write(
    "Ask questions about AI and Machine Learning using "
    "the information in the uploaded research PDFs."
)

@st.cache_resource
def load_rag_components():
    return create_rag_components()

try:
    vectorstore, llm = load_rag_components()
    st.success("Knowledge base connected successfully.")

    question = st.text_input(
        "Enter your question",
        placeholder="What is retrieval-augmented generation (RAG)?",
    )

    if st.button("Ask Question", type="primary"):
        if question.strip():
            with st.spinner("Searching documents and generating an answer..."):
                answer, sources = ask_question(
                    question, vectorstore, llm
                )

            st.subheader("Answer")
            st.markdown(answer)

            st.subheader("Retrieved Sources")
            if sources:
                displayed = set()
                for source in sources:
                    label = (
                        f"{source['source']}, "
                        f"page {source['page']}"
                    )
                    if label not in displayed:
                        st.write(f"- {label}")
                        displayed.add(label)
            else:
                st.info("No relevant sources were retrieved.")
        else:
            st.warning("Please enter a question.")

except Exception as error:
    st.error(f"Unable to start the chatbot: {error}")
    st.info(
        "Check your .env file, Groq API key, and vector store. "
        "If necessary, run build_vectorstore.py first."
    )
