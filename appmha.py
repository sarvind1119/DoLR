#PLEASE check main.py file before going through this

#Import necessary libraries
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from pinecone import ServerlessSpec
#from main import *
from pinecone import Pinecone
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from openai import OpenAI
# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)
from pinecone import ServerlessSpec

cloud = 'aws'
region = 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)

index_name = 'mha'

# get openai api key from platform.openai.com
OPENAI_API_KEY =  os.environ.get('OPENAI_API_KEY')

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY,
    
)

from langchain.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain
index = pc.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    # Initialize the language model with the specified parameters.
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0 )

    # Set up the retriever with the given vector store and search parameters.
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    # Create a retrieval-based QA chain that returns the source documents along with the answers.
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    # Invoke the chain with the provided question and get the response.
    answer = chain.invoke(q)

    # Print the result from the answer.
    print(answer['result'])

    # Print reference information.
    print('Reference:\n')
    # for doc in answer["source_documents"]:
    #     raw_dict = doc.metadata
    #     print("Page number:", raw_dict['page'], "Filename:", raw_dict['source'])
    for x in range(len(answer["source_documents"][0].metadata)):
        raw_dict = answer["source_documents"][x].metadata
        print("Page number:", raw_dict['page'], "Filename:", raw_dict['source'])

    # If needed, return the answer object.
    return answer

# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
import streamlit as st

# Sidebar contents
with st.sidebar:
    st.title('💬 LLM Chat App on Ministry of Home Affairs....')
    st.markdown('''
    ## About
    This GPT helps in answering questions related to Ministry of Home Affairs(using the following Documents)

    [Documents Repository](https://drive.google.com/drive/folders/1YkcVDAXU1gomAzSwY6CK4aRnnzPP3zhJ?usp=drive_link)
    ''')
    # Adding the "Developed by xyz" line in dark green color
    st.markdown('''
    <div style="color: red;">
    Developed by Ashima Goyal
    </div>
    ''', unsafe_allow_html=True)
# Adding the new list with green bullet points
    st.markdown('''
    <div style="color: green;">
    <ul>
        <li>Accessible India Campaign (AIC).pdf</li>
        <li>Advisory on Hon'ble Supreme Court's direction to file FIR in case of Missing Children.pdf</li>
        <li>Advisory on human trafficking in India.pdf</li>
        <li>Advisory on Media Policy of Police-regarding.pdf</li>
        <li>Advisory on the policy for the treatment of terminally ill prisoners.pdf</li>
        <li>Annual Report 2020-21.pdf</li>
        <li>Annual Report 2021-22.pdf</li>
        <li>Annual Report 2022-23.pdf</li>
        <li>AnnualReport_18_19.pdf</li>
        <li>AnnualReport_19_20.pdf</li>
        <li>BM_IIWorksallocation_01092022[1].pdf</li>
        <li>BRDP.pdf</li>
        <li>Comprehensive guidelines regarding service of summons notices.pdf</li>
        <li>Comprehensive report on DM.pdf</li>
        <li>Compulsory Registration of FIR.pdf</li>
        <li>CTCRDivisionworkallocation_08072023.pdf</li>
        <li>Divisionwiseallocation_04072017.pdf</li>
        <li>Guidelines for the preparation of offer list for consideration for appointments to the posts of DG_Additional DG_IG_DIG_SP.pdf</li>
        <li>Guidelines on Constitution and Administration of Preparedness NDRF and SDRF.pdf</li>
        <li>Increasing the number of women in the police forces.pdf</li>
        <li>IS_II_Division_Introduction_12032019.pdf</li>
        <li>J&K Reorganisation Act, 2019.pdf</li>
        <li>Measures to be taken to prevent acid attacks.pdf</li>
        <li>MoU on National Emergency Response System Between Centre and State_UT Government.pdf</li>
        <li>National Disaster Management Plan.pdf</li>
        <li>National Policy on Disaster Management.pdf</li>
        <li>Nationwide Emergency Response System( NERS) Guidelines.pdf</li>
        <li>PDNA handbook.pdf</li>
        <li>Police-II-Intro[1]_1.pdf</li>
        <li>Protection of Life and Property of Senior Citizens.pdf</li>
        <li>Registration of FIR irrespective of territorial jurisdiction and Zero FIR.pdf</li>
        <li>Restriction on powers of remission or commutation in certain cases.pdf</li>
        <li>Scheme Modernisation of Prisons.pdf</li>
        <li>SECURITY RELATED EXPENDITURE Scheme.pdf</li>
        <li>SOP for Post Disaster Need Assessment.pdf</li>
        <li>THE BHARATIYA NAGARIK SURAKSHA SANHITA, 2023.pdf</li>
        <li>THE BHARATIYA NYAYA SANHITA, 2023.pdf</li>
        <li>THE BHARATIYA SAKSHYA ADHINIYAM.pdf</li>
        <li>The Disaster Management Act, 2005.pdf</li>
    </ul>
    </div>
    ''', unsafe_allow_html=True)

    # Add vertical space
    st.markdown('''
    ---

    **In case of suggestions/feedback/Contributions please reach out to:**
    [NIC Training Unit @ nictu@lbsnaa.gov.in]
    ''')


def display_answer(answer):
    st.write("### Query")
    st.write(answer['query'])

    st.write("### Result")
    result = answer['result'].replace('\n', '  \n')  # Ensuring markdown line breaks
    st.markdown(result)

    if "source_documents" in answer:
        st.write("### Reference Documents")
        for i, doc in enumerate(answer["source_documents"], start=1):
            st.write(f"#### Document {i}")
            st.write(f"**Page number:** {doc.metadata['page']}")
            st.write(f"**Source file:** {doc.metadata['source']}")
            content = doc.page_content.replace('\n', '  \n')  # Ensuring markdown line breaks
            st.markdown(content)

def main():
    st.title("Question and Answering App powered by LLM and Pinecone on Ministry of Home Affairs")
    text_input = st.text_input("Ask your query...") 

    if st.button("Ask Query"):
        if len(text_input) > 0:
            answer = ask_and_get_answer(vectorstore, text_input)
            display_answer(answer)

# The main function call
if __name__ == "__main__":
    main()
