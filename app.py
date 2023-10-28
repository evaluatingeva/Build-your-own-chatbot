import os
os.environ['OPENAI_API_KEY'] = "Enter the api key"

import streamlit as st
# Set the title using StreamLit
st.title(' LangChain ')
input_text = st.text_input('Enter Your Text: ') 


from langchain.prompts import PromptTemplate
# setting up the prompt templates
title_template = PromptTemplate(
	input_variables = ['concept'], 
	template='Give me a youtube video title about {concept}'
)

script_template = PromptTemplate(
	input_variables = ['title', 'wikipedia_research'], 
	template='''Give me an attractive youtube video script based on the title {title} 
	while making use of the information and knowledge obtained from the Wikipedia research:{wikipedia_research}'''
)


from langchain.memory import ConversationBufferMemory
# We use the ConversationBufferMemory to can be used to store a history of the conversation between the user and the language model. 
# This information can be used to improve the language model's understanding of the user's intent, and to generate more relevant and coherent responses.

memoryT = ConversationBufferMemory(input_key='concept', memory_key='chat_history')
memoryS = ConversationBufferMemory(input_key='title', memory_key='chat_history')


from langchain.llms import OpenAI
# Importing the large language model OpenAI via langchain
model = OpenAI(temperature=0.6) 

from langchain.chains import LLMChain
chainT = LLMChain(llm=model, prompt=title_template, verbose=True, output_key='title', memory=memoryT)
chainS = LLMChain(llm=model, prompt=script_template, verbose=True, output_key='script', memory=memoryS)


from langchain.utilities import WikipediaAPIWrapper
wikipedia = WikipediaAPIWrapper()

# Display the output if the the user gives an input
if input_text: 
	title = chainT.run(input_text)
	wikipedia_research = wikipedia.run(input_text) 
	script = chainS.run(title=title, wikipedia_research=wikipedia_research)

	st.write(title) 
	st.write(script) 

	with st.expander('Wikipedia-based exploration: '): 
		st.info(wikipedia_research)
