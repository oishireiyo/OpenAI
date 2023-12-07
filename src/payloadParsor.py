import copy

def PayloadParsor(payload: dict, ignore_image: bool=True):
  '''
  OpenAI Python Library内で設定するpayloadを解析する。
  '''
  print(payload)

  _payload = copy.deepcopy(payload)
  for imessage in range(len(_payload['messages'])):
    for icontent in range(len(_payload['messages'][imessage]['content'])):
      if _payload['messages'][imessage]['content'][icontent]['type'] == 'image_url' and ignore_image:
        _payload['messages'][imessage]['content'][icontent]['image_url']['url'] = 'Image information'

  return _payload