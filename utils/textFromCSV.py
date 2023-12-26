import os
import csv

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

def TextFromCSV(csvfile: str):
  text = ''
  with open(csvfile, encoding='shift-jis') as f:
    reader = csv.reader(f)
    for row in reader:
      if len(row) > 0:
        text += row[2]

  return text.split('｡')

def ParseSubtitleCSVreader(csvreader: list):
  subtitle = {'texts': [], 'startsecs': []}
  texts, startframes = [], []
  for row in csvreader:
    if '｡' in row[2]:
      sentences = row[2].split('｡')
      texts.append(sentences[0])
      startframes.append(row[0])

      subtitle['texts'].append(''.join(texts) + '｡')
      subtitle['startsecs'].append(startframes[0])
      if len(sentences) > 1:
        if len(sentences[1]) > 0: texts = [''.join(sentences[1:])]
        else: texts = []
      else:
        texts = []
      startframes = []
    else:
      if len(row[2]) > 0:
        texts.append(row[2])
        startframes.append(row[0])

  return subtitle

def ParseSubtitleCSV(csvfile: str):
  with open(csvfile, encodings='shift-jis') as f:
    csvreader = csv.reader(f)
    return ParseSubtitleCSVreader(csvreader = csvreader)

if __name__ == '__main__':
  texts = TextFromCSV('../assets/helth.csv')