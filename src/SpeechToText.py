import os
import sys
from openai import OpenAI
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
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../utils/')
from payloadParsor import PayloadParsor
from checkOpenAIAPIKeyValid import CheckAPIKeyValid

class SpeechToText():
  def __init__(self, api_key: Union[str, None]=None, model: str='whisper-1') -> None:
    self.OPENAI_API_KEY = api_key if not api_key is None else os.environ['OPENAI_API_KEY']
    self.client = OpenAI(api_key=api_key)
    self.payload = {
      'model': model,
      'file': None, # ここに文字起ししたいファイルを設定する,
      'language': 'ja',
      'response_format': 'text',
      'temperature': 0.0,
      'prompt': '',
    }

  def set_api_key(self, api_key: str) -> bool:
    if CheckAPIKeyValid(api_key=api_key):
      self.client = OpenAI(api_key=api_key)
      return True
    else:
      return False

  '''
  The audio file object to transcribe, in one of these formats:
    flac, mp3, mp4, mpeg, m4a, ogg, wav, webm
  '''
  def set_payload_file(self, filename: str) -> None:
    audio_file = open(filename, 'rb')
    self.payload['file'] = audio_file

  def set_payload_language(self, language: str) -> None:
    self.payload['language'] = language

  def set_payload_response_format(self, response_format: str) -> None:
    self.payload['response_format'] = response_format

  '''
  The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random,
  while lower values like 0.2 will make it more focused and deterministic.
  If set to 0, the model will use log probability to automatically increase the temperature untill certain thresholds are hit.
  '''
  def set_payload_temperature(self, temperature: float=0.0) -> None:
    self.payload['temperature'] = temperature

  def set_payload_prompt(self, prompt: str) -> None:
    self.payload['prompt'] = prompt

  def execute(self):
    transcript = self.client.audio.transcriptions.create(**self.payload)
    print(transcript)

if __name__ == '__main__':
  llm = SpeechToText()
  llm.set_api_key(api_key=os.environ['OPENAI_API_KEY'])
  llm.set_payload_file(filename='../assets/JB-BAN-2304-0139_endroll.mp4')
  llm.set_payload_language(language='ja')
  llm.set_payload_response_format(response_format='srt')
  llm.set_payload_temperature(temperature=0.0)
  llm.set_payload_prompt(prompt='単語帳: 黒ビール、山ねむるエール、岡田')
  llm.execute()