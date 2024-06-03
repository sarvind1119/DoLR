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
from utilsdolr import *

st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ.get('OPENAI_API_KEY'))

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know. ALWAYS give results in bullets points. '""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)


# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()
# Sidebar contents
with st.sidebar:
    st.title('💬 LLM Chat App on Department of Land Resources (DOLR)(Documents)...')

    st.markdown('''
    ## About
    This GPT helps in answering questions related to Department of Land Resources (DOLR)using the following Documents.

    [Documents Repository](https://drive.google.com/drive/folders/1-XT5Exfsspr9Mb5JE9_79M5QwBLlUeG8)
    <div style="color: red;">
    Developed by Shreya Shree
    </div>
    ''', unsafe_allow_html=True)
# Adding the new list with green bullet points
    st.markdown('''
    <div style="color: green;">
    <ul>
        <li>Bhoomi Sampan Certification.pdf</li>
        <li>Bhu-Aadhar_ Unique Land Parcel Identification Nu...pdf</li>
        <li>Cactus.pdf</li>
        <li>Common Guidelines 2008.pdf</li>
        <li>Compensation, Rehabilitation, Resettlement, Development plan rules.pdf</li>
        <li>DILRMP website Intro.pdf</li>
        <li>Draft Operational Guidelines for DILRMP.pdf</li>
        <li>Guidelines for evaluation of preparatory phase of IWMP.pdf</li>
        <li>Guidelines for livelihood component WDC 2.pdf</li>
        <li>Guidelines, Technical Manual & MIS 2018-19.pdf</li>
        <li>Impact Assessment study of Bhoomi Sampan districts.pdf</li>
        <li>Land acquisition (Companies) rules 1963.pdf</li>
        <li>Land Acquisition Act 1894.pdf</li>
        <li>Linkage of e-Court with Land Records_2F Registrati...pdf</li>
        <li>Livelihood Production Systems and Microenterprises - IWMP Operational Guidelines.pdf</li>
        <li>MoRDEnglish_AR2012_13_0.pdf</li>
        <li>National Generic Document Registration System (NG...pdf</li>
        <li>National Rehabilitation and Resettlement Policy 2007 — OCR reqd.pdf</li>
        <li>Registration Act 1908 - Bare Act.pdf</li>
        <li>Revenue Court Case Management System.pdf</li>
        <li>REWARD ToR for Fin Mgmt Expert.pdf</li>
        <li>REWARDS.pdf</li>
        <li>RFCTLARR 2014 ordinance — OCR reqd.pdf</li>
        <li>RFCTLARR 2015 - corrigendum to second amendment.pdf</li>
        <li>RFCTLARR 2015 - second amendment as intro in LS.pdf</li>
        <li>RFCTLARR 2015 Amendment — intro in LS.pdf</li>
        <li>RFCTLARR 2015 Ordinance.pdf</li>
    </ul>
    </div>

    Training on "Wasteland Atlas of India 2019" in progress
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
            response = conversation.predict(input=f"Context://n {context} //n//n Query://n{query}")
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

            if "source_documents" in context:
                st.write("### Reference Documents")
                for i, doc in enumerate(context['source_documents'], start=1):
                    st.write(f"#### Document {i}")
                    st.write(f"**Page number:** {doc.metadata['page']}")
                    st.write(f"**Source file:** {doc.metadata['source']}")
                    content = doc.page_content.replace('//n', '  //n')  # Ensuring markdown line breaks
                    st.markdown(content)




with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')