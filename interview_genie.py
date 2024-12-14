import os
import pandas as pd
import logging
from chromadb import PersistentClient
from chromadb.errors import InvalidCollectionException
from dotenv import load_dotenv
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings  # Example LLM class (replace as needed)
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq  # Example LLM class (replace as needed)
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Constants for configuration
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_NAME = "mixtral-8x7b-32768"
VECTOR_STORE_DIR = "./vectorstore/"
COLLECTION_NAME = "interview_genie_qa"

class InterviewGenieLLM:
    def __init__(self):
        # Initialize the embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

        # Initialize the language model (replace with your actual LLM class)
        self.llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), temperature=0.1, max_tokens=1024)

        # Initialize vector store retriever and other components
        self.retriever = self.get_vectorstore_retriever()
        self.chain = self.create_rag_chain()
        
        # Initialize feedback_chain internally without passing arguments.
        self.feedback_chain = self.create_feedback_chain()

        # Set to keep track of previously asked questions
        self.previous_questions = set()

    def get_vectorstore_retriever(self) -> VectorStoreRetriever:
        """Retrieve the vector store."""
        db = PersistentClient(VECTOR_STORE_DIR)
        try:
            db.get_collection(COLLECTION_NAME)
            retriever = Chroma(
                embedding_function=self.embedding_model,
                collection_name=COLLECTION_NAME,
                persist_directory=VECTOR_STORE_DIR,
            ).as_retriever(search_kwargs={"k": 3})
        except InvalidCollectionException:
            try:
                dataset = pd.read_csv('data/processed_job_postings.csv')  # Replace with actual path
                retriever = self.create_and_store_embeddings(self.process_dataset(dataset)).as_retriever(search_kwargs={"k": 3})
            except FileNotFoundError:
                logging.error("Data file 'data/processed_job_postings.csv' not found.")
                raise
        return retriever

    def process_dataset(self, dataset: pd.DataFrame):
        """Process the dataset into Document instances."""
        documents = []
        for _, row in dataset.iterrows():
            try:
                doc = Document(page_content=f"Job Title: {row['Job Title']}\nDescription: {row['Description']}")
                documents.append(doc)
            except KeyError as e:
                logging.error(f"Missing column in dataset: {e}")
        return documents

    def create_and_store_embeddings(self, documents):
        """Create and store embeddings in the vector store."""
        vectorstore = Chroma.from_documents(
            documents,
            embedding=self.embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=VECTOR_STORE_DIR,
        )
        return vectorstore

    def create_rag_chain(self) -> Runnable:
        """Create a retrieval-augmented generation chain."""
        prompt = ChatPromptTemplate.from_template("""
        You are an intelligent interview assistant AI providing constructive feedback for improvement.
        Context: {context}
        Question: {input}
        Answer:
        """)
        
        document_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)
        retrieval_chain = create_retrieval_chain(self.retriever, document_chain)
        return retrieval_chain

    def create_feedback_chain(self) -> Runnable:
        """Create a feedback chain."""
        prompt = ChatPromptTemplate.from_template("""
        Provide detailed feedback on the following answer to the interview question.
        
        Context: {context}
        Question: {question}
        User Answer: {user_answer}
        
        Feedback:
        """)
        
        feedback_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)
        return feedback_chain

    def generate_interview_question(self, job_posting: str) -> str:
        """Generate a unique interview question based on the job posting."""
        
        while True:
            # Generate a new question using the language model.
            system_message = SystemMessage(content="You are an intelligent assistant generating unique interview questions based on job postings.")
            human_message = HumanMessage(content=f"Generate an interview question based on the following job posting:\n{job_posting}")
            
            response = self.llm.invoke([system_message, human_message])
            question = response.content.strip()
            
            # Check if the question has already been asked.
            if question not in self.previous_questions:
                self.previous_questions.add(question)  # Store the new question.
                return question

    def generate_feedback(self, context_string: str, question_string: str, user_answer: str) -> str:
        """Generate feedback based on context, question, and user answer."""
        
        # Create Document instances for context and question from string inputs.
        context_doc = Document(page_content=context_string)
        question_doc = Document(page_content=question_string)

        # Prepare input data to be passed to the feedback chain.
        input_data = {
            'context': context_doc.page_content,
            'question': question_doc.page_content,
            'user_answer': user_answer
        }

        # Call the feedback chain using the language model to generate dynamic feedback.
        
        system_message = SystemMessage(content="You are an AI providing constructive feedback on interview answers.")
        
        response = self.llm.invoke([system_message] + [HumanMessage(content=f"{key}: {value}") for key, value in input_data.items()])
        
        return response.content.strip()

    def validate_document(self, doc, name: str):
        """Validate that a variable is an instance of Document."""
        
        if not isinstance(doc, Document):
            raise ValueError(f"{name.capitalize()} must be an instance of Document, but received {type(doc).__name__}.")

# Example usage (commented out for clarity; uncomment to use)
# if __name__ == "__main__":
#     interview_genie = InterviewGenieLLM()
#     context_string = "This is a sample context."
#     question_string = "What are your strengths?"
#     user_answer = "I am very organized and detail-oriented."
#     feedback = interview_genie.generate_feedback(context_string, question_string, user_answer)
#     print(feedback)
