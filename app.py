import streamlit as st
from langchain_core.documents import Document
from interview_genie import InterviewGenieLLM

# Initializing the InterviewGenieLLM instance without passing any parameters
interview_genie = InterviewGenieLLM()

# Title of the application
st.title("🎤 Interview Genie - Mock Interview Assistant")

# Input for job posting details
job_posting = st.text_area("Enter the job posting details here:")

# Button to generate an interview question
if st.button("Generate Interview Question"):
    if job_posting:
        # Generate an interview question based on the job posting
        question = interview_genie.generate_interview_question(job_posting)
        # Store the generated question in session state
        st.session_state.generated_question = question
        st.write("Based on the job posting, here is an interview question:")
        st.write(question)
        # Clear the answer input for the next loop
        st.session_state.user_answer = ""
    else:
        st.error("Please enter job posting details.")

# Input for user answer
user_answer = st.text_input("Your Answer:", placeholder="Type your answer here...")
# Store user answer into session state
st.session_state.user_answer = user_answer

# Button to get feedback on the answer
if st.button("Get Feedback"):
    # Ensure that both the generated question and user answer are provided
    if 'generated_question' in st.session_state and st.session_state.generated_question and st.session_state.user_answer:
        # Create a Document object for context
        context = "The candidate is applying for a Data Analyst position where analytical skills and experience with data tools are essential."
        
        # Call the generate_feedback method correctly with string parameters
        feedback = interview_genie.generate_feedback(
            context_string=context, 
            question_string=st.session_state.generated_question, 
            user_answer=st.session_state.user_answer
        )
        
        # Display feedback
        st.write("Feedback:")
        st.write(feedback)
        
        # Automatically generate the next interview question without additional text
        next_question = interview_genie.generate_interview_question(job_posting)
        st.session_state.generated_question = next_question
        
        # Just show the next question without additional context
        st.write(next_question)
        
        # Clear input for the next answer
        st.session_state.user_answer = ""
    else:
        st.error("Please ensure that both the generated question and your answer are provided before requesting feedback.")
