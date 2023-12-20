import os
import cv2
import pprint
import base64
import openai
from openai import OpenAI
import numpy as np
from typing import Union
import json

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# Dummy functions
def get_current_weather(location: str, unit: str='celsius'):
  return json.dumps(
    {'location': 'Tokyo', 'temperature': '10', 'unit': unit}
  )

class FunctionCalling():
  # You can describe functions and have the model intelligently choose to output a JSON object containing arguments to call one or many function.
  # The Chat Completions API does not call the function; instead, the model generates JSON that you can use to call the funtion in your code.
  def __init__(self, api_key: Union[str, None]=None, model: str='gpt-4', max_tokens_per_call: int=200) -> None:
    self.OPENAI_API_KEY=api_key if not api_key is None else os.environ['OPENAI_API_KEY']
    self.client = OpenAI(api_key=self.OPENAI_API_KEY)
    self.payload = {
      'model': model,
      'messages': [], # 質問や画像を並べてモデルに投げる。
      'tools': [], # 実行したい関数を並べる。
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

  # toolsに新しい関数を追加
  def add_tool_entry_as_function(self) -> None:
    self.payload['tools'].append(
      {
        'type': 'function',
        'function': {
          'name': 'get_current_weather',
          'description': 'Get the current weather in a given location.',
          'parameters': {
            'type': 'object',
            'properties': {
              'location': {
                'type': 'string',
                'description': 'The city and state, e.g. San Francisco, CA',
              },
              'unit': {
                'type': 'string',
                'enum': ['celsius', 'fahrenheit'],
                'description': 'The temperature unit to use. Infer this from the users location.',
              },
            },
            'required': ['location'],
          },
        },
      },
    )

  def delete_messages(self) -> None:
    self.payload['messages'] = []

  def delete_content(self, index: int=-1) -> None:
    del self.payload['messages'][index]['content'][index]

  def print_payload(self) -> dict:
    for line in pprint.pformat(self.payload, width=150).split('\n'):
      logger.info(line)
    logger.info('---------------------------------')
    return self.payload

  def execute(self) -> str:
    # The link below will help you figure out which method name is the right one for a given OpenAI model.
    # https://stackoverflow.com/questions/75617865/openai-chatgpt-api-error-invalidrequesterror-unrecognized-request-argument-su
    result = self.client.chat.completions.create(**self.payload)

    logger.info(f'Finish reason: {result.choices[0].finish_reason}')
    logger.info(f'Created: {result.created}')
    logger.info(f'ID: {result.id}')
    logger.info(f'Usage')
    logger.info(f'  Completion tokens: {result.usage.completion_tokens}')
    logger.info(f'  Prompt tokens: {result.usage.prompt_tokens}')
    logger.info(f'  Total tokens: {result.usage.total_tokens}')
    logger.info(f'Model output: {result.choices[0].message.content}')
    logger.info(f'Tool calls function')
    logger.info(f'  Name: {result.choices[0].message.tool_calls[0].function.name}')
    logger.info(f'  Arguments: {result.choices[0].message.tool_calls[0].function.arguments}')

    return result

if __name__ == '__main__':
  gpt4 = FunctionCalling()
  gpt4.add_message_entry_as_specified_role(role='user')
  gpt4.add_text_content(text='What is the weather like in Tokyo')
  gpt4.add_tool_entry_as_function()
  gpt4.print_payload()
  gpt4answer = gpt4.execute()

  pprint.pprint(gpt4answer)