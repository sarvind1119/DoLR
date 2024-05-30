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
from utilsdopng import *

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
    st.title('💬 LLM Chat App on Ministry of Petroleum and Natural Gas...')
    st.markdown('''
    ## About
    This GPT helps in answering questions related to the documents Ministry of Petroleum and Natural Gas

    [Documents Repository](https://drive.google.com/drive/folders/12R75qO425UKg9BdWaS0-Kd-SFRENt3H3)
    ''')
        # Adding the "Developed by xyz" line in dark green color
    st.markdown('''
    <div style="color: red;">
    Developed by Sonika Kumari
    </div>
    ''', unsafe_allow_html=True)
    # Adding the new list with green bullet points
    st.markdown('''
    <div style="color: green;">
    <ul>
        <li>1G-Ethanol.pdf</li>
        <li>Amendments-in-NPR-2018.pdf</li>
        <li>Auto Fuel Policy.pdf</li>
        <li>Auto Fuel Vision and Policy-2025.pdf</li>
        <li>Bharat Petroleum Corporation Limited.pdf</li>
        <li>Bio Diesel.pdf</li>
        <li>CENTRE FOR HIGH TECHNOLOGY (CHT).pdf</li>
        <li>Chennai Petroleum Corporation Limited.pdf</li>
        <li>Compressed Bio Gas.pdf</li>
        <li>Ease of Doing Business.pdf</li>
        <li>Engineers India Limited.pdf</li>
        <li>Ethanol Blended Petrol (EBP) Programme.pdf</li>
        <li>Ethanol_blending_in_India.pdf</li>
        <li>Final_Ethanol_Procurement_Policy.pdf</li>
        <li>Gazette_Notification-10_EBP.pdf</li>
        <li>Guidelines for laying product pipelines.pdf</li>
        <li>Guidelines_for_sale_of_Biodiesel.pdf</li>
        <li>Hindustan Petroleum Corporation Limited.pdf</li>
        <li>History and Evolution.pdf</li>
        <li>Hydrogen Mission.pdf</li>
        <li>Indian Oil Corporation Limited.pdf</li>
        <li>Make In India.pdf</li>
        <li>Mangalore Refinery and Petrochemicals Limited.pdf</li>
        <li>Matters related to MSMEs.pdf</li>
        <li>NATIONAL_POLICY_ON_BIOFUELS-2018.pdf</li>
        <li>Notification_No_-Inclusion_of_IREDA_as_financing_institution_for_scheme_relating_to_augmentation_of_ethanol_production_capacity.pdf</li>
        <li>Notification_No_-New_Scheme_for_extending_financial_assistance_to_sugar_mills_for_enhancement_and_augmentation_of_ethanol_production_capacity.pdf</li>
        <li>Notification_No_-Scheme_for_extending_financial_assistance_to_molasses_based_standalone_distilleries_for_enhancement_and_augmentation_of_ethanol_production.pdf</li>
        <li>Numaligarh Refinery Limited.pdf</li>
        <li>OIL INDUSTRY SAFETY DIRECTORATE.pdf</li>
        <li>PM_JI-VAN_YOJANA.pdf</li>
        <li>Policy-on-Synthetic-Fuels-Committee-Report-March-2024.pdf</li>
        <li>Refinery Capacity.pdf</li>
        <li>Rice-Procurement-Price.pdf</li>
        <li>Scheme_for_extending_financial_assistance_to_project_proponents_for_enhancement_of_their_ethanol_distillation_capacity_or_to_set_up_distilleries_for_producing_1st_G.pdf</li>
        <li>Scheme_for_providing_assistance_to_sugar_mills_for_expenses_on_marketing_costs_-oil_season-of-Sugar_for_Sugar_season_2020-21.pdf</li>
        <li>Second Generation (2G) Ethanol.pdf</li>
        <li>Skill Development.pdf</li>
        <li>Start UP.pdf</li>
        <li>Supply-of-surplus-rice-with-FCI-for-the-production-of-ethanol.pdf</li>
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
                

          