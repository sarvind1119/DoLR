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
    st.title('💬💰 LLM Chat App on Department of Commerce...')
    st.markdown('''
    ## About
    This GPT helps in answering questions related to the documents on Department of Commerce

    [Documents Repository](https://drive.google.com/drive/folders/1olfa2GGSFjeF3XoLnR2Y7N-0lDPptG5y?usp=sharing)
    ''')
        # Adding the "Developed by xyz" line in dark green color
    st.markdown('''
    <div style="color: red;">
    Developed by Mourya & Anup Garg
    </div>
    ''', unsafe_allow_html=True)
    # Adding the new list with green bullet points
    st.markdown('''
    <div style="color: green;">
    <ul>
        <li>12-Ministerial-Conference-Geneva-12-15-June-2022-MC12-outcome-document-Adopted-on-17-June-2022-1.pdf</li>
        <li>Annexure-I-Brief-on-DFTP-Scheme-for-LDC-as-on-16.06.2023-2.pdf</li>
        <li>Annexure-II-DFTP-Scheme-2008.pdf</li>
        <li>Annex_2A_Bis-merged.pdf</li>
        <li>Brief_Glance_on_the_GATS-1.pdf</li>
        <li>Cotton_Tech_Astt._Prg_African_Countries_TAP.pdf</li>
        <li>E-Commerce-Moratorium-Scope-and-Impact (1).pdf</li>
        <li>E-Commerce-Moratorium-Scope-and-Impact.pdf</li>
        <li>Elements_Req_completions_Services_negotiation-1.pdf</li>
        <li>Flag-B1-1.pdf</li>
        <li>Flag-B2.pdf</li>
        <li>Flag-B3.pdf</li>
        <li>IJCEPA_Basic_Agreement.pdf</li>
        <li>Implementing-Agreement.pdf</li>
        <li>Implementing-Procedures.pdf</li>
        <li>IndUAE_Merged.pdf</li>
        <li>List-of-Services-Sectoral-Classification-1.pdf</li>
        <li>Ministerial-Conference-Twelfth-Session-Geneva-12-15-June-2022-Agreement-on-Fisheries-Subsidies-Ministerial-Decision-of-17-June-2022-1.pdf</li>
        <li>Ministerial-Conference-Twelfth-Session-Geneva-12-15-June-2022-Ministerial-Decision-on-the-TRIPS-Agreement-Adopted-on-17-June-2022-1.pdf</li>
        <li>Ministerial-Conference-Twelfth-Session-Geneva-12-15-June-2022-Ministerial-Decision-on-World-Food-Programme-food-purchases-exemption-from-export-prohibitions-or-restrictions-Adopted-on-17-June-2022-1.pdf</li>
        <li>Ministerial-Conference-Twelfth-Session-Geneva-12-15-June-2022-Ministerial-Declaration-on-the-emergency-response-to-food-insecurity-Adopted-on-17-June-2022-1.pdf</li>
        <li>Ministerial-Conference-Twelfth-Session-Geneva-12-15-June-2022-Ministerial-Declaration-on-the-WTO-response-to-the-COVID-19-pandemic-and-preparedness-for-future-pandemics-Adopted-on-17-June-2022-1.pdf</li>
        <li>Ministerial-Conference-Twelfth-Session-Geneva-12-15-June-2022-Work-programme-on-electronic-commerce-Ministerial-Decision-Adopted-on-17-June-2022-1.pdf</li>
        <li>Ministerial-Conference-Twelfth-Session-Geneva-12-15-June-2022-Work-programme-on-small-economies-Ministerial-Decision-Adopted-on-17-June-2022-1.pdf</li>
        <li>Ministerial-Conference-Twelfth-Session-Geneva-12-15-June-2022-WTO-Secretariats-work-in-the-context-of-the-COVID-19-pandemic-Note-by-the-Secretariat-1.pdf</li>
        <li>Ministerial-Conference-Twelfth-SessionGeneva-12-15-June-2022-SPS-declaration-for-the-Twelfth-WTO-Ministerial-Conference-responding-to-modern-SPS-challenges-1 (1).pdf</li>
        <li>Ministerial-Conference-Twelfth-SessionGeneva-12-15-June-2022-SPS-declaration-for-the-Twelfth-WTO-Ministerial-Conference-responding-to-modern-SPS-challenges-1.pdf</li>
        <li>PIB-Release-1-1.pdf</li>
        <li>Practical-Arrangement-on-Information-Exchange-for-implementation-of-Chapter-on-Customs-Procedures.pdf</li>
        <li>Quick-Estimates-April-2024-1.pdf</li>
        <li>SDDS_Data_April-2024-1-1.pdf</li>
        <li>Text-of-GATS.pdf</li>
        <li>W774.pdf</li>
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
                

          