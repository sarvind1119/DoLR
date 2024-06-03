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
from utilscommerce import *

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
    st.title('💬 LLM Chat App on  Ministry of Power documents...')
    st.markdown('''
    ## About
    This GPT helps in answering questions related to document of Ministry of Power.

    [Documents Repository1](https://drive.google.com/drive/folders/1PrS8uaqpAFogLl-lK4IDHDcJ8FA7vjlL?usp=drive_link)
    [Documents Repository2](https://drive.google.com/drive/folders/1I8VMctIyLUNKAC88v7PlZNaA1j7VwmuM)
    ''')
        # Adding the "Developed by xyz" line in dark green color
    st.markdown('''
    <div style="color: red;">
    Developed by Sanjana Sriharsha
    </div>
    ''', unsafe_allow_html=True)
    # Adding the list with green bullet points
    st.markdown('''
    <div style="color: green;">
    <ul>
        <li>Energy-Statistics-India-2023_07022024.pdf</a></li>
        <li>ESN Report-2024_New-21032024.pdf</a></li>
        <li>Manual on Transmission Planning Criteri...</a></li>
        <li>MOP_Annual_Report_Eng_2018-19.pdf</a></li>
        <li>MOP_Annual_Report_Eng_2019-20.pdf</a></li>
        <li>MOP_Annual_Report_Eng_2020-21.pdf</a></li>
        <li>MOP_Annual_Report_Eng_2021-22.pdf</a></li>
        <li>MOP_Annual_Report_Eng_2022-23 (1).pdf</a></li>
        <li>pib.gov.inPressreleaseshare.aspx_PRID=1...</a></li>
        <li>power_sector_at_glance_Feb_2024.pdf</a></li>
        <li>Report-onIndiaRenewableElectricityRoad...</a></li>
        <li>SAARC_framework_agreement_for_energ...</a></li>
        <li>The Electricity Act_2003.pdf</a></li>
        <li>UMPP_Projects_28th_July_2021.pdf</a></li>
        <li style="font-weight: bold; color: black;">below are New Documents added</li>      
        <li>32_Electricity_Timely_Recovery_of_Costs_due_to_Change_in_Law_Rules_2021.pdf</li>
        <li>33_Electricity_Promotion_of_Genration_of_Electricity_from_Must_Run_Power_Plants_Rules_2021.pdf</li>
        <li>35_a_Electricity_Promoting_Renewable_Energy_Through_Green_Energy_Open_Access_Rules_2022.pdf</li>
        <li>Adjudication.pdf</li>
        <li>Amendment_of_RPC.pdf</li>
        <li>Amendment_Tariff_Policy.pdf</li>
        <li>Amendment_Tariff_Policy_2011.pdf</li>
        <li>AppeallateTribunal.pdf</li>
        <li>AppellateTribunal.pdf</li>
        <li>Capacity_Addition_Programme_Beyond_XII_Plan.pdf</li>
        <li>CEA-500GW.pdf</li>
        <li>CERC_Fund.pdf</li>
        <li>Clarification_on_bidding_guidelines.pdf</li>
        <li>clarification_on_tariff_policy.pdf</li>
        <li>CoordinationForum.pdf</li>
        <li>Deemed_Licensestatus_to_MilitaryEngineeringServices.pdf</li>
        <li>DVC_Act_No._XIV_of_1948.pdf</li>
        <li>Electricity Conservation Act.pdf</li>
        <li>Electricity_Amendment Act.pdf</li>
        <li>ForumOfRegulators.pdf</li>
        <li>Guidelines_transmission.pdf</li>
        <li>Hydro_limits.pdf</li>
        <li>JERC_of_Jammu_and_Kashmir_and_UT_of_Ladakh_Preparation_of_Annual_Report_Rules_2023.pdf</li>
        <li>Major_Grid_substations_31st_March_2024.pdf</li>
        <li>National_LoadDispatchCentre.pdf</li>
        <li>Notification_on_CapitalRequirement_And_Creditworthyness.pdf</li>
        <li>Notification_regarding_ Benches_of_APTEL_in_Official_Gazette_May_2012.pdf</li>
        <li>overview_of_Capacity_Addition_during_12th_Plan.pdf</li>
        <li>Power_sector_at_glance_Feb_2024.pdf</li>
        <li>Prepaid_Meters.pdf</li>
        <li>RemovalOfDifficulties.pdf</li>
        <li>Resolution_RegionalPowerCommittes.pdf</li>
        <li>RPC-Second_Amendment_9.5.2008.pdf</li>
        <li>Rural_Electrification_Policy.pdf</li>
        <li>Section 164.pdf</li>
        <li>Sec_126.pdf</li>
        <li>Tariff-policy-28.01.2016-PDF.pdf</li>
        <li>Tariff_Policy-Resolution_28.01.2016-Hindi+English.pdf</li>
        <li>Tariff_Policy-Resolution_Dated_28.01.2016.pdf</li>
        <li>Tariff_Policy.pdf</li>
        <li>The Electricity Act_2003.pdf</li>
        <li>UMPP_Projects_28th_July_2021.pdf</li>
        <li>Works_of_licensees.pdf</li>
        <li>Energy Statistics India 2023_07022024.pdf</li>
        <li>ESN Report-2024_New-21032024.pdf</li>
        <li>Manual on Transmission Planning Criteria 2023.pdf</li>
        <li>MOP_Annual_Report_Eng_2018-19.pdf</li>
        <li>MOP_Annual_Report_Eng_2019-20.pdf</li>
        <li>MOP_Annual_Report_Eng_2020-21.pdf</li>
        <li>MOP_Annual_Report_Eng_2021-22.pdf</li>
        <li>MOP_Annual_Report_Eng_2022-23 (1).pdf</li>
        <li>pib.gov.inPressreleaseshare.aspx_PRID=1992405.pdf</li>
        <li>power_sector_at_glance_Feb_2024.pdf</li>
        <li>Report-onIndiaRenewableElectricityRoadmap2030.pdf</li>
        <li>SAARC_framework_agreement_for_energy_cooperation_electricity.pdf</li>
        <li>The Electricity Act_2003.pdf</li>
        <li>UMPP_Projects_28th_July_2021.pdf</li>
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
                

          