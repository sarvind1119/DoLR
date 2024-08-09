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
import csv
import os
import streamlit as st
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
    st.title('💬💫 LLM Chat App on World Bank documents on projects done in India..')
    st.markdown('''
    ## About
    This GPT helps in answering questions related to Project documents done in India by worldbank

    [Documents Repository](https://drive.google.com/drive/folders/1QdUlbJaviFV3WM-GueRYhY2NmPZ2YxaK)
    ''')
        # Adding the "Developed by xyz" line in dark green color
    st.markdown('''
    <div style="color: red;">
    Developed by Naresh Akunuri
    </div>
    ''', unsafe_allow_html=True)
    # Adding the new list with green bullet points
    st.markdown('''
    <div style="color: green;">
    <ul>
        <li>AR_Ministry_of_Textiles_2020-21_Eng.pdf</li>
        <li>AR_Ministry of Textiles_2021-22_Eng.pdf</li>
        <li>AR_MoT_2019-20_English.pdf</li>
        <li>Central Silk Board-AC-I-and-RULES-Book.pdf</li>
        <li>CSR Report-2017(MoT).pdf</li>
        <li>Demand for Grants 2022-2023.pdf</li>
        <li>Enhancing_Export_Competitiveness_Textile_Sector_03042018.pdf</li>
        <li>FDI Scenario in Indian Textiles Sector - A Study Report.pdf</li>
        <li>Final Report - Employment Study.pdf</li>
        <li>Final Report on Direct and Indirect Taxes.pdf</li>
        <li>Garment Study - Annexures - 26.02.2018.pdf</li>
        <li>IIFT-J - Textile Project Report 20th March (1).pdf</li>
        <li>Innovations in Textile and Apparel Industry.pdf</li>
        <li>Jute Packaging Materials (Compulsory Use in Packing Commodities) Act 1987.pdf</li>
        <li>MOT Annual Report 2022-23 (English).pdf</li>
        <li>national textiles policy 2000.pdf</li>
        <li>nift_act.pdf</li>
        <li>PM_MITRA_Guidelines.pdf</li>
        <li>Report_SITP_26Dec2016.pdf</li>
        <li>Textiles-AnnualReport2018-2019(English).pdf</li>
        <li>THE JUTE COMPANIES (NATIONALISATION), ACT, 1980.pdf</li>
        <li>THE TEXTILE UNDERTAKINGS (NATIONALISATION)ACT, 1995.pdf</li>
        <li>THE TEXTILES COMMITTEE ACT, 1963.pdf</li>
        <li>To Promote Growth of Man Made Fibre.pdf</li>
        <li>Vision Strategy Action Plan for Indian Textile Sector-July15.pdf</li>
        <li>Wazir Advisors-FTA study-Final Report.pdf</li>
    </ul>
    </div>
    ''', unsafe_allow_html=True)
    # Add vertical space
    st.markdown('''
    ---

    **In case of suggestions/feedback/Contributions 🤔 please reach out to:**
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


def save_to_csv(data, filename="conversation_log.csv"):
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(["User Input", "Refined Query", "Result"])  # Write header only if file doesn't exist

        writer.writerow(data)

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

            # Save the query, refined query, and result to CSV
            save_to_csv([query, refined_query, response])


with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
                

          