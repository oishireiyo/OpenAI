import pprint
import numpy as np
from sentence_transformers import SentenceTransformer

def TextEmbedding(texts: list[str]):
  model = SentenceTransformer('pkshatech/simcse-ja-bert-base-clcmlp')
  embeddings = model.encode(texts)
  return embeddings

def CosSimilarity(vec1, vec2):
  return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def CosSimilarities(vec1s, vec2):
  cosines = []
  for vec1 in vec1s:
    cosine = CosSimilarity(vec1=vec1, vec2=vec2)
    cosines.append(cosine)
  return cosines

if __name__ == '__main__':
  from textFromCSV import TextFromCSV, ParseSubtitleCSV
  subtitles = ParseSubtitleCSV('../assets/helth.csv')
  embeddings = TextEmbedding(texts=subtitles['texts'])

  openaitexts = [
    '日本の厚生労働省は新しい運動指針を発表し、成人に毎日少なくとも60分の歩行と週に数回の筋力トレーニングを推奨している。',
    '指針は、定期的な身体活動が生活習慣病や早死にのリスクを減らすことを強調し、特に高齢者には毎日少なくとも40分の歩行と筋力トレーニングを勧めている、日本の平均的な成人は1日7時間座っており、深刻な健康問題につながる可能性がある。',
    '30分ごとに立つなど、定期的に中断することで、長時間の座位に伴うリスクを軽減できる。',
    '正しい座位姿勢の実演や簡単なエクササイズを行うことで、座っていても筋肉活動やエネルギー消費を維持できることが示された。',
  ]
  openaiembeddings = TextEmbedding(texts=openaitexts)

  cosines = CosSimilarities(vec1s=embeddings, vec2=openaiembeddings[0])

  textcosines = [[text, cosine] for text, cosine in zip(subtitles['texts'], cosines)]
  textcosines = sorted(textcosines, key=lambda x: x[1])
  sortedtexts = [text for text, _ in textcosines]

  print(sortedtexts)