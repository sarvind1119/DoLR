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
from utilslaw import *

st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ.get('OPENAI_API_KEY'))

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""ALWAYS give results in bullets points if the answer is of multiple lines. 
                                                                if the user asks the answer in tabular format give a nice html table in the answer.
                                                                Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know. '""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)


# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()
# Sidebar contents
with st.sidebar:
    st.title('💬 LLM Chat App on  Judgements By High Court on POSCO...')
    st.markdown('''
    ## About
    This GPT helps in answering questions related to document of Judgements By High Court on POSCO.

    [Documents Repository1](https://drive.google.com/drive/folders/1PrS8uaqpAFogLl-lK4IDHDcJ8FA7vjlL?usp=drive_link)
    [Documents Repository2](https://drive.google.com/drive/folders/1I8VMctIyLUNKAC88v7PlZNaA1j7VwmuM)
    ''')
        # Adding the "Developed by xyz" line in dark green color
    st.markdown('''
    <div style="color: red;">
    Developed by Anju Mam
    </div>
    ''', unsafe_allow_html=True)
    # Adding the new list with green bullet points
    st.markdown('''
    <div style="color: green;">
    <ul>
        <li>Ataur_Mridha_vs_The_State_01122020__BDADBDAD20202309211158441COM772183.pdf</li>
        <li>Gajraj_Ramabhai_Hajani_vs_State_of_Gujarat_2010202GJ2020260321164247123COM144007.pdf</li>
        <li>In_Reference_vs_Sachin_Kumar_Singhraha_03032016__MMP2016010422165414179COM116132.pdf</li>
        <li>Jitendra_vs_The_State_of_Madhya_Pradesh_31032022__MP2022150622165100366COM982841.pdf</li>
        <li>Krishna_Murti_vs_State_of_NCT_of_Delhi_05042021__DDE202127092117185318COM630986.pdf</li>
        <li>MANU-DE-0358-2017-Del20220625114113.pdf</li>
        <li>MANU-DE-0574-2020-Del20220625113928.pdf</li>
        <li>MANU-DE-0574-2020-Del20220625113954.pdf</li>
        <li>MANU-DE-1588-2020-Del20220625113756.pdf</li>
        <li>MANU-DE-1769-2020-Del20220625113904.pdf</li>
        <li>MANU-DE-1972-2018-Del20220625114153.pdf</li>
        <li>MANU-DE-3470-2017-Del20220625114042.pdf</li>
        <li>MANU-DE-4924-2017-Del20220625114020.pdf</li>
        <li>MANU-GH-0514-2021-GH20220625114217.pdf</li>
        <li>MANU-HP-0103-2019-HP20220625111124.pdf</li>
        <li>MANU-HP-1029-2020-HP20220625114743.pdf</li>
        <li>MANU-KA-3119-2021-Kar20220625114244.pdf</li>
        <li>MANU-MH-1433-2019-Bom20220625113540.pdf</li>
        <li>MANU-MH-1434-2019-Bom20220625113607.pdf</li>
        <li>MANU-MH-2554-2021-Bom20220625113655.pdf</li>
        <li>MANU-MH-3021-2018-Bom20220625113511.pdf</li>
        <li>MANU-MH-3771-2021-Bom20220625113628.pdf</li>
        <li>MANU-MP-0592-2020-MP20220625114515.pdf</li>
        <li>MANU-MP-0983-2021-MP20220625113040.pdf</li>
        <li>MANU-OR-0309-2021-OR20220625114714.pdf</li>
        <li>MANU-PH-0406-2022-PH20220625112549.pdf</li>
        <li>MANU-PH-0407-2022-PH20220625112618.pdf</li>
        <li>MANU-RH-1043-2021-RAJ20220625113144.pdf</li>
        <li>MANU-TN-1558-2022-MAD20220625112656.pdf</li>
        <li>MANU-TN-2177-2022-MAD20220625112637.pdf</li>
        <li>MANU-TN-5810-2021-MAD20220625114859.pdf</li>
        <li>MANU-TN-6914-2021-MAD20220625114836.pdf</li>
        <li>MANU-TN-7339-2021-MAD20220625114413.pdf</li>
        <li>MANU-TR-0228-2020-TR20220625114541.pdf</li>
        <li>MANU-TR-0244-2020-TR20220625114613.pdf</li>
        <li>MANU-TR-0250-2021-TR20220625114641.pdf</li>
        <li>MANU-UP-3256-2021-ALL20220625111827.pdf</li>
        <li>MANU-UP-3378-2021-ALL20220625111800.pdf</li>
        <li>MANU-WB-0583-2022-CAL20220625112033.pdf</li>
        <li>MANU-WB-1127-2021-CAL20220625111919.pdf</li>
        <li>MANU-WB-1128-2021-CAL20220625111854.pdf</li>
        <li>Mohammad_Juned_vs_State_of_Madhya_Pradesh_04102014MP2014090516154017116COM719500.pdf</li>
        <li>NBDA Annual Report 2016_17.pdf</li>
        <li>NBDA Annual Report 2018_19.pdf</li>
        <li>NHRC Annual Reports_2017_18.pdf</li>
        <li>nhrc report 2015_101to200.pdf</li>
        <li>OneHundredFifthReport _1.pdf</li>
        <li>Pramod_Yadav_vs_The_State_of_Madhya_Pradesh_and_OrMP2021200721162309197COM875849.pdf</li>
        <li>Rajendran_vs_State_23122016__MADHCTN2016190417150743135COM170941.pdf</li>
        <li>Ramji_vs_State_of_Haryana_and_Ors__09102020__PHHCPH2020090221155944210COM653183.pdf</li>
        <li>Ram_Dayal_and_Ors_vs_State_of_UP_and_Ors_14122018_UP2018150519163237301COM932678.pdf</li>
        <li>RIT_Foundation_and_Ors_vs_The_Union_of_India_and_ODE20221205221757381COM477081.pdf</li>
        <li>State_of_Madhya_Pradesh_vs_Ravi_and_Ors_29092016__MP2016080217160223144COM576739.pdf</li>
        <li>Subrata_Pradhan_and_Ors_vs_State_of_West_Bengal_anWB2022020622171934471COM33448.pdf</li>
        <li>Takla_vs_State_of_UP_and_Ors_20072021__ALLHCUP2021131221162113312COM583890.pdf</li>
        <li>Vikas_vs_State__19102020__DELHCDE202022122116083919COM825823.pdf</li>
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
                

          