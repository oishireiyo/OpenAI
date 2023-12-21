import os
import sys
import pprint
from openai import OpenAI
import numpy as np
from typing import Union

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# Handmade modules
sys.path.append(os.pardir + '/utils/')
from payloadParsor import PayloadParsor
from textCosmetics import TextCosmetics

class TextGeneration():
  def __init__(self, api_key: Union[str, None]=None, model: str='gpt-4-1106-preview', max_tokens_per_call: int=200) -> None:
    self.OPENAI_API_KEY = api_key if not api_key is None else os.environ['OPENAI_API_KEY']
    self.client = OpenAI(api_key=self.OPENAI_API_KEY)
    self.payload = {
      'model': model,
      'messages': [],
      'max_tokens': max_tokens_per_call,
    }

  # messagesに新しいエントリーを追加
  def add_message_entry_as_specified_role(self, role: str) -> None:
    self.payload['messages'].append({'role': role, 'content': []})

  # テキストを最新のmessages内に追加
  def add_text_content(self, text: str):
    self.payload['messages'][-1]['content'].append(
      {'type': 'text', 'text': text}
    )

  def add_message_entry_as_specified_role_with_text_content(self, role: str, text: str) -> None:
    self.add_message_entry_as_specified_role(role=role)
    self.add_text_content(text=text)

  def delete_messages(self) -> None:
    self.payload['messages'] = []

  def delete_content(self, index: int=-1) -> None:
    del self.payload['messages'][-1]['content'][index]

  def print_payload(self) -> dict:
    for line in pprint.pformat(self.payload, width=150).split('\n'):
      logger.info(line)
    logger.info('-' * 50)
    return self.payload
  
  def execute(self) -> dict:
    result = self.client.chat.completions.create(**self.payload)

    logger.info(f'Finish reason: {result.choices[0].finish_reason}')
    logger.info(f'Created: {result.created}')
    logger.info(f'ID: {result.id}')
    logger.info(f'Usage')
    logger.info(f'  Completion tokens: {result.usage.completion_tokens}')
    logger.info(f'  Prompt tokens: {result.usage.prompt_tokens}')
    logger.info(f'  Total tokens: {result.usage.total_tokens}')
    logger.info(f'Model output: {result.choices[0].message.content}')

    return result
  
if __name__ == '__main__':
  try:
    import sys
    sys.path.append('../../DeepLAPI')
    from translator import DeepLTranslator
    translator = DeepLTranslator()
  except Exception as e:
    logger.warning(e)

  llm = TextGeneration()
  llm.add_message_entry_as_specified_role_with_text_content(
    role='system',
    text=translator.translate(
      text='あなたは質問に対して簡潔に回答するアシスタントです。',
      source_lang='JA',
      target_lang='EN-US',
    ) if 'translator' in locals() else
    'You are a helpful assistant designed to answer questions in a concise manner.'
  )
  # llm.add_message_entry_as_specified_role_with_text_content(
  #   role='user',
  #   text=translator.translate(
  #     text='2021-2022年シーズンにNBAを制したチームと、その本拠地はどこですか?',
  #     source_lang='JA',
  #     target_lang='EN-US',
  #   ) if 'translator' in locals() else
  #   'Who won the NBA title in 2021-2022 season and where is its home town?'
  # )
  # llm.add_message_entry_as_specified_role_with_text_content(
  #   role='assistant',
  #   text=translator.translate(
  #     text='ゴールデンステート・ウォリアーズ、カリフォルニア州サンフランシスコ。',
  #     source_lang='JA',
  #     target_lang='EN-US',
  #   ) if 'translator' in locals() else
  #   'Golden State Warriors, San Francisco, California.'
  # )
  # llm.add_message_entry_as_specified_role_with_text_content(
  #   role='user',
  #   text=translator.translate(
  #     text='2020-2021年シーズンにNBAを制したチームと、その本拠地はどこですか?',
  #     source_lang='JA',
  #     target_lang='EN-US',
  #   ) if 'translator' in locals() else
  #   'Who won the NBA title in 2020-2021 season and where is its home town?'
  # )

  textfile = open('../assets/transcription.txt', 'r')
  llm.add_message_entry_as_specified_role_with_text_content(
    role='user',
    text=translator.translate(
      text=TextCosmetics(
        text=f'以下はあるニュース番組の一場面を文字起こししたものです。キーワードをカンマ区切りで10個抽出してください。\
          "{textfile.read(-1)}"'
      ),
      source_lang='JA',
      target_lang='EN-US',
    ) if 'translator' in locals() else 'Are you happy?'
  )

  _ = llm.print_payload()
  llmanswer = llm.execute()

  translated_llmanswer = translator.translate(
    text=llmanswer.choices[0].message.content,
    source_lang='EN',
    target_lang='JA',
  )

  logger.info(translated_llmanswer)