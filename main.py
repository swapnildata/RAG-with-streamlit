import streamlit as st
from langchain.llms import OpenAI
import os
openai_key = os.environ.get('OPENAI_API_KEY')
llm=OpenAI(openai_api_key=openai_key)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

st.title("Basic GenAI Application")
template_string=st.text_input("Hi user. I am powerfull LLM. Ask me anything.")
temp=st.slider('select the value for temprature',min_value=0.0, max_value=2.0, step=0.1)
chat=ChatOpenAI(openai_api_key=openai_key,temperature=temp,model="gpt-3.5-turbo")
prompt_template=ChatPromptTemplate.from_template(template_string)
content=prompt_template.format_messages()
LLM_response = chat(content)
st.write(LLM_response.content)
uploaded_file=st.file_uploader("Choose a text file :",type="txt")
if uploaded_file is not None:
    file_content = uploaded_file.read().decode("utf-8")
    st.write(file_content)