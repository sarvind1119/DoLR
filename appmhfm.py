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
from utilsmhfm import *

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
    st.title('💬 LLM Chat App on Ministry of Health & Family Welfare...')
    st.markdown('''
    ## About
    This GPT helps in answering questions related to the documents Ministry of Health & Family Welfare

    [Documents Repository](https://drive.google.com/drive/folders/11ts4fc9E1DpO10kxXlbe2c2PCmTeKsKk?usp=sharing)
    ''')
        # Adding the "Developed by xyz" line in dark green color
    st.markdown('''
    <div style="color: red;">
    Developed by Anand Malhotra
    </div>
    ''', unsafe_allow_html=True)
    # Adding the new list with green bullet points
    st.markdown('''
    <div style="color: green;">
    <ul>
        <li>01-SDH_DH_IPHS_Guidelines-2022.pdf</li>
        <li>02-CHC_IPHS_Guidelines-2022.pdf</li>
        <li>03_PHC_IPHS_Guidelines-2022.pdf</li>
        <li>04-SHC_HWC_UHWC_IPHS_Guidelines-2022.pdf</li>
        <li>Ayushman Bharat H&WC.pdf</li>
        <li>Comprehensive National Nutrion Survey 2016-18.pdf</li>
        <li>DHR_Annual Report_20-21.pdf</li>
        <li>DHR_Annual Report_21-22.pdf</li>
        <li>DHR_Annual Report_22-23.pdf</li>
        <li>DoH&FW_Annual Report_20-21.pdf</li>
        <li>DoH&FW_Annual Report_21-22.pdf</li>
        <li>Draft_NUHM_framework-2023.pdf</li>
        <li>Drugs and Cosmetics Act 1940.pdf</li>
        <li>EMR-EHR_Standards_for_India_as_notified_by_MOHFW_2016_0.pdf</li>
        <li>Final NPRD, 2021.pdf</li>
        <li>FSSAI Act 2006.pdf</li>
        <li>G20_HMM_Outcome_Document_and_Chair_Summary.pdf</li>
        <li>Governance Reforms in Medical Education 2014-2020.pdf</li>
        <li>Health System Strengthening_Conditionality Report of States 2018-19.pdf</li>
        <li>HealthandFamilyWelfarestatisticsinIndia201920.pdf</li>
        <li>HMIS 2020-21 & 2021-22.pdf</li>
        <li>India TB Report 2023.pdf</li>
        <li>LASI_India_Report_2020_compressed.pdf</li>
        <li>Mental Healthcare Act, 2017.pdf</li>
        <li>National Digital Health Blueprint.pdf</li>
        <li>National Health Policy 2017.pdf</li>
        <li>National Policy for Containment of AMR.pdf</li>
        <li>NFHS-5_INDIA_REPORT.pdf</li>
        <li>NHM_2009-14.pdf</li>
        <li>NRHM Framework for Implementation 2005-12.pdf</li>
        <li>RHS 2019-20_2.pdf</li>
        <li>RHS 2021-22_2.pdf</li>
        <li>Towards_Universal_Health_Coverage_HWC_Compendium_2018-20.pdf</li>
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
                

          