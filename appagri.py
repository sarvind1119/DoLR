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
from utilsagri import *
import os
st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ.get('OPENAI_API_KEY'))

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""ALWAYS give results in bullets point if response is of multiple lines.
                                                                If user asks answer in other format specifically then answer in that format only.
                                                                Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I think i don't have context of that question'  '""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)




# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()
# Sidebar contents
with st.sidebar:
    st.title('💬 LLM Chat App on Ministry of Agriculture and Farmers welfare ...')
    st.markdown('''
    ## About
    This GPT helps in answering questions related to the documents Ministry of Agriculture and Farmers welfare 

    [Documents Repository](https://drive.google.com/drive/folders/1qG50cpiwSsYNE7QzWHkg4FxQjkeN3Pew?usp=drive_link)
    ''')
        # Adding the "Developed by xyz" line in dark green color
    st.markdown('''
    <div style="color: red;">
    Developed by Sneha Arugula
    </div>
    ''', unsafe_allow_html=True)
    # Adding the new list with green bullet points
    st.markdown('''
    <div style="color: green;">
    <ul>
        <li>ACABC-Revised-Guideline-2018.pdf</li>
        <li>Brief introduction about the SMAF scheme_0.pdf</li>
        <li>Cooperation division.pdf</li>
        <li>crisis management plan for drought.pdf</li>
        <li>CRMGuidelines2023-24.pdf</li>
        <li>Crop insurance brief.pdf</li>
        <li>disaster assistance.pdf</li>
        <li>divisions of Ministry.pdf</li>
        <li>Draft Guidelines for bio-stimulands.pdf</li>
        <li>Drought manual.pdf</li>
        <li>Final Compendium 4-10-2018.pdf</li>
        <li>Final_guidelines NMSA.pdf</li>
        <li>Guidelines_for_IMCT_2023.pdf</li>
        <li>Guideline_MIF03082018.pdf</li>
        <li>MOVCDNERGUIDELINES.pdf</li>
        <li>NADMP.pdf</li>
        <li>national policy for farmers.pdf</li>
        <li>NeGP-A.pdf</li>
        <li>NFSM.pdf</li>
        <li>NMSA_Guidelines_English_1.pdf</li>
        <li>NPMCR_1.pdf</li>
        <li>NRM division.pdf</li>
        <li>OG-PMKMY_ENG.pdf</li>
        <li>oilseeds.pdf</li>
        <li>OperationalGuidelinesofPM-KISANScheme.pdf</li>
        <li>Op_Guidelines_of_AGROFORESTY_RKVY.pdf</li>
        <li>Organisational History.pdf</li>
        <li>PKVY_Guidelines.pdf</li>
        <li>PMFBY darft operational guidelines.pdf</li>
        <li>PMFBY operational guidelines.pdf</li>
        <li>Ready-reckoner-PP-I.pdf</li>
        <li>rkvy_inro.pdf</li>
        <li>SMAM_Guidelines__Revised_in_2020_Migrant_Labours_.pdf</li>
        <li>SMSP13.05.2014_2.pdf</li>
        <li>Soil health card.pdf</li>
        <li>Sub Mission on Agroforestry operational guidelines_0.pdf</li>
        <li>Vermicompost Production Unit.pdf</li>
        <li><b style="color:black;">ACTS & Annual Reports</b></li>
        <li>46_of_1968_0.pdf</li>
        <li>Act-2020-Contract-Farming-27-09-2020-5.pdf</li>
        <li>Annual Report 2019-20.pdf</li>
        <li>Annual Report-2015-16.pdf</li>
        <li>Annual-Report 2011-12.pdf</li>
        <li>Annual-Report-2010-11.pdf</li>
        <li>Annual-Report-2013-14.pdf</li>
        <li>annual-report-2020-21.pdf</li>
        <li>annual-report-2021-22.pdf</li>
        <li>annual_report_english_2022_23.pdf</li>
        <li>Annual_rpt_201617.pdf</li>
        <li>apmc-16.pdf</li>
        <li>AR 2017-18.pdf</li>
        <li>AR-2014-15.pdf</li>
        <li>AR_2018-19_Final_Print.pdf</li>
        <li>Citizen_Charter_DAC_2017.pdf</li>
        <li>contract_farm2018-12.pdf</li>
        <li>data_0_0.pdf</li>
        <li>DIP_Act_0_0.pdf</li>
        <li>Dispute Resolution Rules Gazette notification_0.pdf</li>
        <li>efinalorg-21.pdf</li>
        <li>Essential_Commodity_Act_1955(Amend_upto_1986).pdf</li>
        <li>e_Gazettee_The Farmers_Produce_Trade_and_Commerce_Promotion_and_Facilitation_ Rules_2020_1-3.pdf</li>
        <li>FAQs on farmers agreement ordinance,_2.pdf</li>
        <li>FAQTradeCommerce08102020_1_4.pdf</li>
        <li>Farming Agreement Final Act _3.pdf</li>
        <li>FinalDraftRules2007-17.pdf</li>
        <li>gmr1988-19.pdf</li>
        <li>Guidelines farm services Act, 2020 -converted_3.pdf</li>
        <li>handbookworkallo_2018.pdf</li>
        <li>Indian_Plant_Quarantine_Act2023_0.pdf</li>
        <li>itbill2000.pdf</li>
        <li>MSCSAct2002.pdf</li>
        <li>NCDC General Regulation.pdf</li>
        <li>NCDCACT-2002.pdf</li>
        <li>NEW_POLICY_NPSD.pdf</li>
        <li>OOMF 2022-23.pdf</li>
        <li>Seed_Act_1966.pdf</li>
        <li>SEED_Ammendment_Act_1972.pdf</li>
        <li>Seed_Control_Order_1983.pdf</li>
        <li>Seed_Rule_1968.pdf</li>

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