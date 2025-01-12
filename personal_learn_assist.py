from langchain_google_genai import GoogleGenerativeAI
from pypdf import PdfReader
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import LLMChain

#streamlit run personal_learn_assist.py
load_dotenv()

llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key=os.getenv("GOOGLE_API_KEY"))

def pdf_reader(pdf_files, num):
    if not pdf_files:
        return "No file is uploaded"
    text = ""
    for pdf_file in pdf_files:
        try:
            pdf_loader = PdfReader(pdf_file)
            for page in pdf_loader.pages:
                text += page.extract_text()
        except Exception as e:
            return f"Error reading PDF: {e}"
    text = text.replace('\t', ' ') 
    
    template = """
    You have to generate {num} questions. 
    The text is: {text}.
    
    The format must be:
    1. What is the capital of France?
    2. What is the best known food in Paris?
    Like this a list must be returned.

    Ensure:
    1. The questions are at most 2 lines long.
    """
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)
    data = chain.run({"text": text, "num": num})
    return [data,text]

def answer(selected,text):
    template = """
    Using the text content:
    {text}
    Write the answers for the question {selected}.
    The answer can be brief or detailed, depending upon the question.
    Make sure the answer is not incomplete.
    Format of answer:
    Answer: The best food available in Paris is churos.
    """
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)
    data = chain.run({"text": text, "selected":selected})
    return data


def main():
    st.title("Question Generator Based on PDF Uploaded")
    st.subheader("Let's see if you can answer them")
    num_inp = st.number_input("How many questions do you want?", max_value=100, min_value=1)
    pdf = st.file_uploader("Upload Your file here", type=["pdf"])
    
    if "data" not in st.session_state:
        st.session_state.data = []
    if "selected_questions" not in st.session_state:
        st.session_state.selected_questions = []
    if "text" not in st.session_state:
        st.session_state.text = ""

    if pdf is not None:
        generate = st.button("Generate questions")
        if generate:
            data_dict = pdf_reader([pdf], num_inp)
            st.session_state.data = data_dict[0].split("\n")
            st.session_state.text = data_dict[1]
            st.write("Generated Questions:")
            st.write(st.session_state.data)

    if st.session_state.data:
        selected_options = st.multiselect(
            "Select questions",
            st.session_state.data,
        )
        st.session_state.selected_questions = selected_options
        st.write("You selected:", selected_options)

    if st.session_state.selected_questions:
        get_ans = st.button("Get Answers")
        if get_ans:
            for selected in st.session_state.selected_questions:
                st.write(f"**Question:** {selected}")
                answer_text = answer(selected, st.session_state.text)
                st.write(f"**Answer:** {answer_text}")
        

if __name__ == "__main__":
    main()
