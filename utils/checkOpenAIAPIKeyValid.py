import os
from openai import OpenAI

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

def CheckAPIKeyValid(api_key: str) -> bool:
  try:
    _client = OpenAI(api_key=api_key)
    _client.completions.create(
      model='davinci',
      prompt='Say this is a test.',
      max_tokens=5,
    )
  except Exception as e:
    logger.error(e)
    return False
  else:
    logger.info('Given API key is valid.')
    return True