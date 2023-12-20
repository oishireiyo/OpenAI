import os
import cv2
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

# The images API provides three methods for interacting with images:
# 1. Creating images from scratch based on a text prompt (DALL-E 3 and DALL-E 2)
# 2. Creating edited versions of images by having the model replace some areas of a pre-existing image, based on a new text prompt (DALL-E 2 only)
# 3. Creating variations of an existing image (DALL-E 2 only)
class TextPrompt():
  # With the release of DALL-E 3, the model now takes in the default prompt provided and automatically re-write if to safety reasons, and to add more detail.
  # While it is not currently possible to disable this feature, you can use prompting to get outputs closer to your requested image by adding the following to your prompt:
  # `I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:`.
  # The updated prompt is visible in the `revised_prompt` field of the data response object.
  def __init__(self, api_key: Union[str, None]=None, model: str='dall-e-3', max_tokens_per_call: int=200) -> None:
    self.OPENAI_API_KEY = api_key if not api_key is None else os.environ['OPENAI_API_KEY']
    self.client = OpenAI(api_key=self.OPENAI_API_KEY)
    self.payload = {
      'model': model,
      'prompt': '', # ここに生成したいイメージの詳細を記述する。
      'size': '1024x1024',
      'quality': 'standard',
      'n': 1,
      'response_format': 'url',
    }

  def set_payload_prompt(self, prompt: str) -> None:
    self.payload['prompt'] = prompt

  def set_payload_size(self, size: str='1024x1024') -> None:
    # When using DALL-E 3, image can have a size of 1024x1024, 1024x1792 or 1792x1024 pixels.
    accepted_sizes = {
      'dall-e-3': ['1024x1024', '1024x1792', '1792x1024'],
      'dall-e-2': ['256x256', '512x512', '1024x1024'],
    }
    if size in accepted_sizes[self.payload['model']]:
      self.payload['size'] = size

  def set_payload_quality(self, quality: str='standard') -> None:
    # By default, images are generated at `standard` quality, but when using DALL-E 3 you can set `quality: "hd"` for enhanced detail.
    accepted_qualities = {
      'dall-e-3': ['standard', 'hd'],
      'dall-e-2': ['standard'],
    }
    if quality in accepted_qualities[self.payload['model']]:
      self.payload['quality'] = quality

  def set_payload_n(self, n: int=1) -> None:
    # You can request 1 image at a time with DALL-E 3 (request more by making parallel requests) or up to 10 images at a time using DALL-E 2 with the n parameter.
    if self.payload['model'] == 'dall-e-2' and (n >= 0 and n <= 10):
      self.payload['n'] = n

  def set_payload_response_format(self, response_format: str='url'):
    # The format in which the generated images are returned. Must be one of `url` or `b64_json`.
    accepted_response_format = ['url', 'b64_json']
    if response_format in accepted_response_format:
      self.payload['response_format'] = response_format

  def set_payload_style(self, style: str='vivid'):
    # The style of the generated images. Must be one of the `vivid` or `natural`. This param is only supported for `dall-e-3`.
    if self.payload['model'] == 'dall-e-3':
      self.payload['style'] = style

  def print_payload(self) -> dict:
    logger.info(self.payload)
    return self.payload
  
  def execute(self) -> str:
    result = self.client.images.generate(**self.payload)

    print(result.data[0].b64_json)
    print(result.data[0].url)
    print(result.data[0].revised_prompt)
    print(len(result.data))

    return result.data

class Edits():
  # The image edits endpoint allows you to edit or extend and image by uploading an image and mask indicating which areas should be replaced.
  # The transparent areas of the mask indicate where the imahe should be edited, and the prompt should describe the full new imahe, not just the erased area.
  # This endpoint can enable experiences like the editor in our DALL-E preview app.
  def __init__(self, api_key: Union[str, None]=None, max_tokens_per_call: int=200) -> None:
    self.OPENAI_API_KEY = api_key if not api_key is None else os.environ['OPENAI_API_KEY']
    self.client = OpenAI(api_key=self.OPENAI_API_KEY)
    self.payload = {
      'model': 'dall-e-2',
      'prompt': '', # ここに生成したいイメージの詳細を記述する。
      'image': None,
      'mask': None,
      'n': 1,
      'size': '1024x1024',
    }

  def set_payload_prompt(self, prompt: str) -> None:
    self.payload.prompt = prompt

  def set_payload_image(self, image: str) -> None:
    # The uploaded image and mask must both be square PNG images less than 4MB in size,
    # and also must have the save ddimensions as each other.
    # The non-transparent areas of the mask are not used when generating the output,
    # so they don't nevessarily need to match the original image.
    self.payload.image = open(image, 'rb')

  def set_payload_mask(self, mask: str) -> None:
    self.payload.mask = open(mask, 'rb')

  def execute(self) -> str:
    result = self.client.images.edit(**self.payload)

    return result.data

class Variations():
  # The image variations endpoint allows you to generate a variation of a given image.
  def __init__(self, api_key: Union[str, None]=None, max_tokens_per_call: int=200) -> None:
    self.OPENAI_API_KEY = api_key if not api_key is None else os.environ['OPENAI_API_KEY']
    self.client = OpenAI(api_key=self.OPENAI_API_KEY)
    self.payload = {
      'model': 'dall-e-2',
      'image': None,
      'n': 2,
      'size': '1024x1024',
    }

  def set_payload_image(self, image: str) -> None:
    self.payload.image = open(image, 'rb')

  def execute(self) -> str:
    result = self.client.images.create_variation(**self.payload)

    return result.data

if __name__ == '__main__':
  try:
    import sys
    sys.path.append('../../DeepLAPI/')
    from translator import DeepLTranslator
    translator = DeepLTranslator()
  except Exception as e:
    logger.warning(e)

  prompter = TextPrompt()
  prompter.set_payload_prompt(
    prompt=translator.translate(
      text='10代の若々しいアジア人女性の画像。',
      source_lang='JA',
      target_lang='EN-US',
    ) if 'translator' in locals() else
    'A photograph of a bunch of cats in Tokyo.',
  )
  prompter.set_payload_quality(
    quality='hd',
  )
  prompter.set_payload_style(style='vivid')
  prompter.print_payload()
  _ = prompter.execute()