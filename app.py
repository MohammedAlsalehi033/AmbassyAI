import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()



# Load the CSV file
loader = CSVLoader(file_path="./passport_application_qa.csv")
documents = loader.load()

model = SentenceTransformer('all-MiniLM-L6-v2')

document_texts = [doc.page_content for doc in documents]

document_embeddings = model.encode(document_texts)

# Save embeddings and documents
with open('/path/to/save/documents.pkl', 'wb') as f:
    pickle.dump(document_texts, f)

with open('/path/to/save/embeddings.pkl', 'wb') as f:
    pickle.dump(document_embeddings, f)

# Create FAISS index
d = document_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(document_embeddings)

# Save FAISS index
faiss.write_index(index, '/path/to/save/faiss_index.index')

def retrieve_info(query):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=5)
    results = [document_texts[i] for i in I[0]]
    return " ".join(results)

# Initialize the LLM with the API key
llm = ChatOpenAI(temperature=0.4, model="gpt-3.5-turbo")

template = """
You are a highly knowledgeable and efficient embassy helper chatbot.
You work at the Yemen Embassy in Islamabad.
You will receive a query from a user, and you will provide the best response 
that follows all the rules and best practices below:

1. The response should closely follow the established best practices in terms of length, tone of voice, logical structure, and detailed information.
2. If the best practice is irrelevant to the query, try to mimic the style of the best practice to formulate the response.
3. You should only respond to queries related to embassy services. If you do not know the answer or the query is outside the scope of embassy services, you should politely apologize and indicate that you do not have the information.
4. Your response will be directly sent to the user, so it should be formatted accordingly.
5. Your response should be according to the language of the user.
6. Respond in Arabic if not specified.

Context: {context}

Below is a query I received from the user:
{message}

Please write the best response.
"""

# Example usage in your script
prompt_template = PromptTemplate(template=template, input_variables=["message", "context"])

chain = LLMChain(llm=llm, prompt=prompt_template)

def generate_response(message):
    context = retrieve_info(message)
    response = chain.run(message=message, context=context)
    return response

def main():
    st.set_page_config(
        page_title="Digital ambassador", page_icon="📄")

    st.header("Digital ambassador 📄")
    message = st.text_area("Customer message")

    if message:
        st.write("Generating Response...")

        result = generate_response(message)

        st.info(result)

if __name__ == '__main__':
    main()
