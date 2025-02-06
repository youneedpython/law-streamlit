from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from config import answer_examples


## 환경변수 읽어오기 #######################################################
load_dotenv()
# pinecone_api_key = os.environ.get('PINECONE_API_KEY')

def get_retrieval():
  ## embedding ################################################
  embedding = OpenAIEmbeddings(model='text-embedding-3-large')

  ## vector databse: 파인콘에 저장된 인덱스 정보 가져오기 #####
  ## 파인콘에 생성한 인덱스명
  index_name = 'law-index'

  ## 파인콘에 생성된 인덱스 정보 가져오기
  database = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding,
  )

  ## 파인콘에 있는 함수로 similarity search 수행
  retriever = database.as_retriever(search_kwargs={'k': 4})
  return retriever


def get_llm(model='gpt-4o'):
  ## ChatGPT의 LLM 모델 지정
  llm = ChatOpenAI(model=model)
  return llm


def get_dictionary_chain():
  ## dictionary 참고하여, prompt 변경 
  dictionary = ['전세 사기 당한 사람 -> 전세사기피해자']
  llm = get_llm()

  prompt = ChatPromptTemplate.from_template(f'''
      사용자의 질문을 보고, 우리 사전을 참고하여 사용자의 질문을 변경해주세요.
      만약 변경할 필요가 없다고 판단되면, 사용자의 질문을 변경하지 않아도 됩니다.
      그리고 사용자가 한 질문을 그대로 리턴해주세요.
      질문에 대한 답변을 모르면, '죄송합니다. 다시 질문해주세요.'라고 답변해 주세요.
      사전: {dictionary}

      질문: {{question}} 
  ''')

  ## llm을 거쳐서 dictionary를 참고하여 질문 새로 생성
  dictionary_chain = prompt | llm | StrOutputParser()
  return dictionary_chain


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
  if session_id not in store:
      store[session_id] = ChatMessageHistory()
  return store[session_id]


def get_history_retriever():
  llm = get_llm()  
  retriever = get_retrieval()

  # qa_chain = RetrievalQA.from_chain_type(
  #   llm,
  #   retriever=retriever,
  #   chain_type_kwargs={'prompt': prompt}
  # )

  contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
  )

  contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
  )

  history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
  )

  return  history_aware_retriever


def get_rag_chain():
  ## RetrievalQA를 이용한 LLM 질문 ################################
  # ## LangSmith API KEY 설정
  # LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

  # ## prompt 모델 지정
  # prompt = hub.pull('rlm/rag-prompt', api_key=LANGCHAIN_API_KEY)

  # ## RetrievalQA 객체에 llm, collection, prompt 설정
  llm = get_llm()

  ## few-shot ##################################################### 
  ## https://python.langchain.com/v0.2/docs/how_to/few_shot_examples/#creating-the-example-set
  example_prompt = PromptTemplate.from_template("Question: {input}\n{answer}")

  few_shot_prompt = FewShotPromptTemplate(
    examples=answer_examples,
    example_prompt=example_prompt,
    suffix="Question: {question}",
    input_variables=["question"],
  )  

  ## FewShotPromptTemplate을 문자열로 변환
  formatted_few_shot_prompt = few_shot_prompt.format(question="{input}")

  system_prompt = (
    "당신은 전세사기 피해자법 전문가입니다. 사용자의 전세사기 피해자관련 질문에 답변해주세요."
    "아래에 제공된 문서를 활용해서 답변해주시고,"
    "답변을 알 수 없다면, 모른다고 답변해주세요."
    "답변을 제공할 때는 '전세사기피해자법 XX조에 따르면'으로 출처 표시해주세요."
    "\n\n"
    "{context}"
  )

  qa_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", system_prompt),
          ("assistant", formatted_few_shot_prompt), ## 문자열로 변환한 프롬프트 추가
          MessagesPlaceholder("chat_history"),
          ("human", "{input}"),
      ]
  )

  question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
  history_aware_retriever = get_history_retriever()
  rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

  conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
  ).pick('answer')

  return conversational_rag_chain


## [AI Message 함수 정의] +++++++++++++++++++++++++++++++++++++++++++++++++++
def get_ai_response(user_message):
  dictionary_chain = get_dictionary_chain()
  # qa_chain = get_qa_chain()
  rag_chain = get_rag_chain()

  law_chain = {'input': dictionary_chain} | rag_chain
  ai_response = law_chain.stream(
     input={'question': user_message},
     config={"configurable": {"session_id": "abc123"}},
     )
  
  return ai_response
  # return ai_message['answer']
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
