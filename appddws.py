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
from utilsddws import *

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
# Sidebar contents
with st.sidebar:
    st.title('💬 LLM Chat App on Department of Drinking Water and Sanitation....')
    st.markdown('''
    ## About
    This GPT helps in answering questions related to Department of Drinking Water and Sanitation(using the following Documents)

    [Documents Repository](https://drive.google.com/drive/folders/1YkcVDAXU1gomAzSwY6CK4aRnnzPP3zhJ?usp=drive_link)
    ''')
    # Adding the "Developed by xyz" line in dark green color
    st.markdown('''
    <div style="color: red;">
    Developed by Vikas Ruhela
    </div>
    ''', unsafe_allow_html=True)
   # Adding the new list with green bullet points
    st.markdown('''
    <div style="color: green;">
    <ul>
        <li>DDWS_annual-report-2022-23-eng.pdf</li>
        <li>DDWS_Behavioural-change-communication-strategy.pdf</li>
        <li>DDWS_capacity-building-plan.pdf</li>
        <li>DDWS_Compendium-of-behavioural-best-practices.pdf</li>
        <li>DDWS_JJS-2023-june-2023-bulletin.pdf</li>
        <li>DDWS_RWPF-Annual-Yearbook-2023.pdf</li>
        <li>JJM-Brochure-Year-2024.pdf</li>
        <li>100 DAYS PLAN-Suggestions.pdf</li>
        <li>IEC_Manual_English.pdf</li>
        <li>Karnataka_Udupi_MRF Unit.pdf</li>
        <li>Kerala-Plastic_and_other_NBW.pdf</li>
        <li>PWM_Manual_English_InnerPages.pdf</li>
        <li>Rural_Sanitation_Strategy_Report.pdf</li>
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