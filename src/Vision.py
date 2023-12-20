import os
import cv2
import pprint
import base64
import openai
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
from payloadParsor import PayloadParsor

class Vision():
  # Messages must be an array of message object, where each object has a role (either "system", "user" or "assistant") and content.
  # Conversations can be short as one message or many back and forth turns.

  # Typically, a conversation is formatted with a system message first, followed by alternating user and assistant messages.

  # The system message helps set the begavior the assistant.
  # For example, you can modify the personaluty of the assistant or provide specific instructions about how it should behave throughout the conversation.
  # However note that the system message is optional and the model's behavior without a system message is likely to be similar to using a generic message such as "You are helphul assistant".

  # The user messages provide requests or comments for the assistant to respond to.
  # Assistant messages store previous assistant responses, but can also be written by you to give examples of desired behavior.
  def __init__(self, api_key: Union[str, None]=None, model: str='gpt-4-vision-preview', max_tokens_per_call: int=200) -> None:
    # GPT-4 with vision
    self.OPENAI_API_KEY = api_key if not api_key is None else os.environ['OPENAI_API_KEY']
    self.client = OpenAI(api_key=self.OPENAI_API_KEY)
    self.payload = {
      'model': model,
      'messages': [], # ここに質問や画像などを並べてモデルに投げる。
      'max_tokens': max_tokens_per_call,
    }

  def set_api_key(self, api_key: str) -> bool:
    openai.api_key = api_key
    try:
      _ = openai.Completion.create(
        model="davinci",
        messages=[
          {'role': 'user', 'content': 'Who are you?'}
        ],
        max_tokens=5,
      )
    except:
      logger.error('Given API key is not valid.')
      return False
    else:
      self.client = OpenAI(api_key=api_key)
      return True

  # messagesに新しいエントリーを追加
  def add_message_entry_as_specified_role(self, role: str) -> None:
    self.payload['messages'].append({'role': role, 'content': []})

  # テキストを最新のmessages内に追加
  def add_text_content(self, text: str):
    self.payload['messages'][-1]['content'].append(
      {'type': 'text', 'text': text}
    )

  # 画像を最新のmessages内に追加
  # By controlling the `detail` parameter, which has three options, `low`, `high` or `auto`,
  # you have control over how the model processes the image and generates its textual understanding.
  # By default, the model with use the `auto` setting which will look at the image input size and decide if it should use the `low` or `high` setting.
  def encode_image_path(self, pathimage: str) -> str:
    with open(pathimage, 'rb') as f:
      return base64.b64encode(f.read()).decode('utf-8')

  def encode_image_array(self, ndarrayimage: np.ndarray) -> str:
    ret, buffer = cv2.imencode('.png', ndarrayimage)
    if ret:
      return base64.b64encode(buffer).decode('utf-8')

  def add_urlimage_content(self, urlimage: str, detail: str='auto') -> None:
    self.payload['messages'][-1]['content'].append(
      {'type': 'image_url', 'image_url': {'url': urlimage}, 'detail': detail}
    )

  def add_b64image_content(self, b64image: str, detail: str='auto') -> None:
    self.payload['messages'][-1]['content'].append(
      {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64image}', 'detail': detail}}
    )

  def add_pathimage_content(self, pathimage: str, detail: str='auto') -> None:
    b64image = self.encode_image_path(pathimage=pathimage)
    self.add_b64image_content(b64image=b64image, detail=detail)

  # 動画を最新のmessages内に追加
  def add_pathvideo_content(self, pathvideo: str, increment: int, detail: str='auto') -> None:
    video = cv2.VideoCapture(pathvideo)
    for iframe in range(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)), increment):
      video.set(cv2.CAP_PROP_POS_FRAMES, iframe)
      ret, frame = video.read()
      if ret:
        b64image = self.encode_image_array(ndarrayimage=frame)
        self.add_b64image_content(b64image=b64image, detail=detail)
      else:
        logger.error('Frame No.%d of input video could not be read.')

  def add_content(self, contents: dict, as_type: str) -> None:
    if as_type == 'text':
      self.add_text_content(input_message_in_ja=contents['message'], glossary_name=contents['glossary'])
    if as_type == 'urlimage':
      self.add_urlimage_content(url=contents['url'], detail=contents['details'])
    if as_type == 'b64image':
      self.add_b64image_content(b64image=contents['b64image'], detail=contents['details'])

  def delete_messages(self) -> None:
    self.payload['messages'] = []

  def delete_content(self, index: int=-1) -> None:
    del self.payload['messages'][-1]['content'][index]

  def print_payload(self) -> dict:
    for line in pprint.pformat(PayloadParsor(self.payload), width=100).split('\n'):
      logger.info(line)
    logger.info('---------------------------------')
    return self.payload

  # Every response will include a `finish_reason`. The possible values for `finish_reason` are:
  # - `stop`: API returned complete message, or a message terminated by one of the stop sequences provided via the stop parameter.
  # - `length`: Incomplete model output due to `max_tokens` parameter or token limit.
  # - `function_call`: The model decided to call a function.
  # - `content_filter`: Omitted content due to a flag from our content filters.
  # - `null`: API response still in progress or incomplete.
  def execute(self) -> str:
    result = self.client.chat.completions.create(**self.payload)

    logger.info(f'Finish reason: {result.choices[0].finish_reason}')
    logger.info(f'Created: {result.created}')
    logger.info(f'ID: {result.id}')
    logger.info(f'Usage')
    logger.info(f'  Completion tokens: {result.usage.completion_tokens}')
    logger.info(f'  Prompt tokens: {result.usage.prompt_tokens}')
    logger.info(f'  Total tokens: {result.usage.total_tokens}')
    logger.info(f'Model output: {result.choices[0].message.content}')

    return result.choices[0].message.content

if __name__ == '__main__':
  try:
    import sys
    sys.path.append('../../DeepLAPI/')
    from translator import DeepLTranslator
    translator = DeepLTranslator()
  except Exception as e:
    logger.warning(e)

  gpt4v=Vision()
  gpt4v.add_message_entry_as_specified_role(role='system')
  gpt4v.add_text_content(
    text=translator.translate(
      text='質問に対して簡潔に回答してください。',
      source_lang='JA',
      target_lang='EN-US',
    ) if 'translator' in locals() else
    'You are a helpful assistant designed to output a sentence as short as possible which describe the following contents.',
  )

  # gpt4v.add_message_entry_as_specified_role(role='user')
  # gpt4v.add_text_content(text='What do you see in this image?')
  # gpt4v.add_pathimage_content(pathimage='../assets/ahoaho.jpeg', detail='row')

  # gpt4v.add_message_entry_as_specified_role(role='assistant')
  # gpt4v.add_text_content(text='dog, balls, leash, grass')

  gpt4v.add_message_entry_as_specified_role(role='user')
  gpt4v.add_text_content(
    text=translator.translate(
      text='以下の画像は「地球温暖化で寿司が食べられなくなる！？」というタイトルの動画から時系列順に抽出したものです。各々の画像の説明は行わずに、動画の内容を予想してください。',
      source_lang='JA',
      target_lang='EN-US',
    ) if 'translator' in locals() else
    'What is the context of these images?',
  )
  # gpt4v.add_pathvideo_content(pathvideo='../assets/GenerativeAI.mp4', increment=90, detail='low')
  gpt4v.add_pathvideo_content(pathvideo='../assets/sushi.mp4', increment=90, detail='low')

  gpt4v.print_payload()
  gpt4vanswer = gpt4v.execute()

  text=translator.translate(
    text=gpt4vanswer,
    source_lang='EN',
    target_lang='JA',
  )

  logger.info(text)