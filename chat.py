import streamlit as st 
from llm import get_ai_response

##########################################################
## 전세사기 피해 당한 사람이 받을 수 있는 금융 지원은?
##########################################################

st.set_page_config(page_title='전세사기피해자자 상담 챗봇', page_icon='🍀')
st.title('전세사기피해자 상담 챗봇')
st.caption('전세사기피해 법률 상담해드립니다.')


if 'message_list' not in st.session_state:
  st.session_state.message_list = []

print(f'before: {st.session_state.message_list}')

for message in st.session_state.message_list:
  with st.chat_message(message['role']):
    st.write(message['content'])

if user_question := st.chat_input(placeholder='전세사기 피해와 관련된 질문을 작성해 주세요.'):
  with st.chat_message('user'):
    st.write(user_question)
  st.session_state.message_list.append({'role': 'user', 'content': user_question})

  with st.spinner('답변을 생성하는 중입니다.'):
    ai_response = get_ai_response(user_question)

    with st.chat_message('ai'):
      ai_message = st.write_stream(ai_response)
      st.session_state.message_list.append({'role': 'ai', 'content': ai_message})


print(f'after: {st.session_state.message_list}')
