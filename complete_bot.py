import streamlit as st
from langchain_core.prompts import PromptTemplate
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import os


DB_FAISS_PATH = "vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    # st.write("Vectorstore loaded successfully!")
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN,
                      "max_token": "512"}
    )
    return llm


def format_source_documents(source_documents):
    """Make the source documents clean, readable string."""
    if not source_documents:
        return "No source documents available."
    
    formatted_output = "### Source Documents\n"
    for i, doc in enumerate(source_documents, 1):
        # Extract content and metadata (if available)
        content = doc.page_content if hasattr(doc, 'page_content') else "No content available"
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        
        # Truncate content if too long for better readability
        content_preview = (content[:300] + "...") if len(content) > 300 else content
        
        formatted_output += f"##### *Document {i}:*\n"
        formatted_output += f"- **Content Preview**: {content_preview}\n\n"
        if metadata:
            formatted_output += f"- **Metadata**: {metadata}\n"
        formatted_output += "\n"
    
    return formatted_output


def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        custom_prompt_template = """
            Use the pieces of information provided in the context to answer user's question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Don't provide anything out of the given context.

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please.
        """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")
        llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)


        try:
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed t0 load vector store.")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            formatted_sources = format_source_documents(source_documents)
            result_to_show = f"{result}\n\n{formatted_sources}"

            # result_to_show = result + "\n\n" + str(source_documents)
            # response="Hi, I am MediBot!"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistent', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__=="__main__":
    main()