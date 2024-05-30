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
from utilsdpiit import *

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
    st.title('💬 LLM Chat App on Department for Promotion of Industry and Internal Trade...')
    st.markdown('''
    ## About
    This GPT helps in answering questions related to the documents of Department for Promotion of Industry and Internal Trade

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
        <li>amendment_ADR_Act_1951_0.pdf</li>
        <li>approval clearances required for new projects.pdf</li>
        <li>ArmsRule_28May2017_3.pdf</li>
        <li>boilersAct_1923_14September2023.pdf</li>
        <li>Channel of submission.pdf</li>
        <li>Citizen_Charter_2023-24.pdf</li>
        <li>Clarification_AfterSale_Repair_Services.pdf</li>
        <li>clarification_contract_manufacturing_13122019.pdf</li>
        <li>clarification_FDI policy_Facility_Sharing_Group_Companies.pdf</li>
        <li>clarification_FPRT_20June2018.pdf</li>
        <li>Clarification On FDI Policy On Single Brand Retail Trading.pdf</li>
        <li>copy_Right_Act_14July2016.pdf</li>
        <li>copy_RightAmendment_14July2016.pdf</li>
        <li>design_act.pdf</li>
        <li>Digital-Media-Clarification-Scanned-160x42020.pdf</li>
        <li>Ease-DoingBusiness-Booklet-10March2021.pdf</li>
        <li>Explosive_Act_1884_0.pdf</li>
        <li>Fact Sheet December 2023_1.pdf</li>
        <li>FAQApproved-FDIPolicy-19November2020.pdf</li>
        <li>FDI-PolicyCircular-2020-29October2020_0.pdf</li>
        <li>format_fdi_policy10052013.pdf</li>
        <li>Gazette_Notification_of_RTI_Amendment_Act_2019_23june2022.pdf</li>
        <li>GI_Act.pdf</li>
        <li>Guidelines-Special_Assistance_to_States_for_Capital_Investments_03102023.pdf</li>
        <li>IEM FAQ.pdf</li>
        <li>inductionMaterial_19September2023.pdf</li>
        <li>Industrial License Services.pdf</li>
        <li>industries_Amendment_Act_2016.pdf</li>
        <li>InflammableSubstance_Act_1952_03July2018.pdf</li>
        <li>Investment_Opportunities_in_India.pdf</li>
        <li>InvestmentPromotion-Reforms-10March2021.pdf</li>
        <li>IPR-EnforcementToolkit-19January2021.pdf</li>
        <li>IPR-Reforms-11June2021.pdf</li>
        <li>JanVishwas_Act_21August2023.pdf</li>
        <li>JKDFC_AnnualReport_2022-23_05February2024.pdf</li>
        <li>LogisticsDivision_QuartelyReport_19January2024.pdf</li>
        <li>ManufacturingPolicy-Reforms-11June2021.pdf</li>
        <li>NationalIndustrialCorridor-Programme-11June2021.pdf</li>
        <li>NationalLogisticsPolicy_2022_21September2023.pdf</li>
        <li>NIC-2008.pdf</li>
        <li>NID_Act_2014_19August2014_0.pdf</li>
        <li>Promotion-StartUps-11June2021.pdf</li>
        <li>PublicProcurement-Reforms-11June2021.pdf</li>
        <li>Reforms-BoilersDivision-11June2021.pdf</li>
        <li>Revised_InternshipScheme_19October2023.pdf</li>
        <li>ROLE AND FUNCTIONS - DPIIT.pdf</li>
        <li>RTI act.pdf</li>
        <li>SOP for Data Management on PM GatiShakti NMP (1) (1).pdf</li>
        <li>THE_NATIONAL_INSTITUTE_OF_DESIGN_AMENDMENT_ACT_2019_03July2020.pdf</li>
        <li>tm_Amendment_Act_2010.pdf</li>
        <li>TMRAct_New.pdf</li>
        <li>winning_moves.pdf</li>
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
                

          