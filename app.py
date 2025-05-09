# import streamlit as st
# import time
# from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

# import os
# os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# def user_input(user_question):
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chatHistory = response['chat_history']
#     for i, message in enumerate(st.session_state.chatHistory):
#         if i%2==0:
#             st.write("User:",message.content)
#         else:
#             st.write("Reply:",message.content)

# def main():
#     st.set_page_config(page_title="Information Retrieval")
#     st.header("Information-Retrieval-Sytem-using-GenAI")
#     user_question = st.text_input("Ask a Question from the PDF!")

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chatHistory" not in st.session_state:
#         st.session_state.chatHistory = None
#     if user_question:
#         user_input(user_question)


#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your pdf files and click on the submit & process button", accept_multiple_files = True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
                
#                 raw_text = get_pdf_text(pdf_docs)
#                 print(raw_text)
#                 text_chunks = get_text_chunks(raw_text)
#                 print(text_chunks)
#                 vector_store = get_vector_store(text_chunks)
#                 print(vector_store)
#                 st.session_state.conversation = get_conversational_chain(vector_store)

#                 st.success("Done")
# if __name__ == "__main__":
#     main()

import streamlit as st
import time
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain
import os

# Disable file watcher to prevent performance issues on some environments
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Function to handle user input and display conversation
def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']

    for i, message in enumerate(st.session_state.chatHistory):
        with st.chat_message("user" if i % 2 == 0 else "assistant"):
            st.markdown(message.content)

# Main function
def main():
    st.set_page_config(page_title="📄 PDF Q&A System", page_icon="🔍", layout="wide")
    st.title("🔍 Information Retrieval System using GenAI")

    # Sidebar for file upload
    with st.sidebar:
        st.markdown("### 📂 Upload PDF")
        pdf_docs = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("📥 Submit & Process"):
            if pdf_docs:
                with st.spinner("🔄 Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("✅ PDFs Processed Successfully!")
            else:
                st.warning("⚠️ Please upload at least one PDF.")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []

    # Input section
    with st.container():
        st.markdown("### 💬 Ask a question about your uploaded PDF:")
        user_question = st.text_input("Type your question here...", placeholder="e.g., What is the summary of the document?")
        if user_question and st.session_state.conversation:
            user_input(user_question)
        elif user_question:
            st.warning("⚠️ Please process the PDF first before asking questions.")

if __name__ == "__main__":
    main()
