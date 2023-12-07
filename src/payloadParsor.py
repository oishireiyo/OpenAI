import copy

def PayloadParsor(payload: dict, ignore_image: bool=True):
  '''
  OpenAI Python Library内で設定するpayloadを解析する。
  '''
  _payload = copy.deepcopy(payload)
  for imessage in len(_payload['messages']):
    for icontent in len(_payload['messages'][imessage]['content']):
      if _payload['messages'][imessage]['content'][icontent]['type'] == 'image_url' and ignore_image:
        _payload['messages'][imessage]['content'][icontent]['image_url']['url'] = 'Image information'

  return _payload