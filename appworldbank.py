from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils_worldbank import *

st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ.get('OPENAI_API_KEY'))

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Try to answer in bullets points if the answer is in multiple lines. 
                                                                Answer the question as truthfully as possible using the provided context,

and if the answer is not contained within the text below, say 'I think i don't have that context. '""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)




# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()
# Sidebar contents
with st.sidebar:
    st.title('💬 LLM Chat App on Projects in India by World bank...')
    st.markdown('''
    ## About
    This GPT helps in answering questions Projects in India by World bank

    [Documents Repository](https://drive.google.com/drive/folders/1Oyf1oSZFnQ4y7ttRBBjs3i5JWcQh0S8k)
    ''')
        # Adding the "Developed by xyz" line in dark green color
    st.markdown('''
    <div style="color: red;">
    Developed by Parth Gupta
    </div>
    ''', unsafe_allow_html=True)
    # Adding the list with green bullet points
    st.markdown('''
<div style="color: green;">
<ul>
    <li>CFPC_2017_FINAL_RELEASED_28.8.17_0.pdf</li>
    <li>FDI-PolicyCircular-2020-29October2020_1.pdf</li>
    <li>FDI_Circular_012011_31March2011 5_0.pdf</li>
    <li>FDI_Circular_01_2012 3_0.pdf</li>
    <li>FDI_Circular_01_2013 2(1).pdf</li>
    <li>FDI_Circular_02of2010 6_0.pdf</li>
    <li>fdi_circular_1_2010 7_1.pdf</li>
    <li>FDI_Circular_2014 1_0.pdf</li>
    <li>FDI_Circular_2015 1_0_0.pdf</li>
    <li>FDI_Circular_2016(2).pdf</li>
    <li>gypsum based building rules.pdf</li>
    <li>industrial license.pdf</li>
    <li>Courses directory.pdf</li>
    <li>Dissertation Draft by SKARYA-4902 01.02.24.pdf</li>
    <li>Dissertation Draft by SKARYA-4902 05.03.24.pdf</li>
    <li>Dissertation Draft by SKARYA-4902 10.03.24.pdf</li>
    <li>Dissertation Draft by SKARYA-4902 13.03.24.pdf</li>
    <li>Dissertation Draft by SKARYA-4902 15.03.24.pdf</li>
    <li>Dissertation Draft by SKARYA-4902 23.02.24.pdf</li>
    <li>Final-Dissertation Draft by SKARYA-4902-19.03.24.pdf</li>
</ul>
</div>
''', unsafe_allow_html=True)

    # Add vertical space
    st.markdown('''
    ---

    **In case of suggestions/feedback/Contributions please reach out to:**
    [NIC Training Unit @ nictu@lbsnaa.gov.in]
    ''')


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = ask_and_get_answer(vectorstore,refined_query)
            # print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

            if "source_documents" in context:
                st.write("### Reference Documents")
                for i, doc in enumerate(context['source_documents'], start=1):
                    st.write(f"#### Document {i}")
                    st.write(f"**Page number:** {doc.metadata['page']}")
                    st.write(f"**Source file:** {doc.metadata['source']}")
                    content = doc.page_content.replace('\n', '  \n')  # Ensuring markdown line breaks
                    st.markdown(content)




with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
                

          
