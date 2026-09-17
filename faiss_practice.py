# pip install faiss-cpu

import numpy as np
import time
import faiss
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 1. 기존 방식 (브루트포스) - 지금 API가 쓰는 방식
documents = [
    "타이타닉은 1912년 4월 15일 빙산과 충돌해 침몰한 영국의 여객선이다.",
    "파이썬은 1991년 귀도 반 로섬이 개발한 프로그래밍 언어이다.",
    "BERT는 구글이 2018년에 발표한 자연어처리 모델이다.",
    "김치는 발효 채소를 이용한 한국의 전통 음식이다.",
    "RAG는 검색과 생성을 결합한 자연어처리 기법이다.",
    "딥러닝은 인공신경망을 여러 층으로 쌓아 학습하는 머신러닝의 한 분야이다.",
]

doc_embeddings = embed_model.encode(documents)
print("임베딩 shape:", doc_embeddings.shape)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_bruteforce(query, top_k=2):
    # 모든 문서를 순회하며 유사도 계산 (문서 수에 비례해 느려짐)
    query_vec = embed_model.encode(query)
    scores = [cosine_sim(query_vec, doc_vec) for doc_vec in doc_embeddings]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(documents[i], float(scores[i])) for i in top_indices]

# 2. FAISS 인덱스 구축
# FAISS는 내적이나 L2 거리로 검색함
# 코사인 유사도를 쓰려면: 벡터를 정규화(normalize) -> 내적 = 코사인 유사도
# 정규화하면 벡터 길이가 1이 되어 내적이 곧 코사인 값이 됨
dimension = doc_embeddings.shape[1]    # 768차원

# 벡터 정규화 (코사인 유사도를 위해 필수)
doc_embeddings_norm = doc_embeddings.copy().astype('float32')
faiss.normalize_L2(doc_embeddings_norm)

# IndexFlatIP: Inner Product 기반 인덱스
# "Flat"은 압축 없이 전체를 저장한다는 뜻 -> 정확하지만 메모리를 많이 씀
# 소규모 데이터 (수만 개 이하)에는 이게 적합함
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings_norm)
print("인덱스에 저장된 벡터 수:", index.ntotal)

def search_faiss(query, top_k=2):
    # FAISS 인덱스로 검색
    query_vec = embed_model.encode(query).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vec)    # 질문 벡터도 동일하게 정규화

    # search는 유사도 배열, 인덱스 배열을 반환
    scores, indices = index.search(query_vec, top_k)

    return [(documents[i], float(s)) for i, s in zip(indices[0], scores[0])]

# 3. 두 방식 결과 비교 (정확도가 같은지 확인)
query = "RAG가 뭐야?"

print("\n[브루트포스]")
for doc, score in search_bruteforce(query):
    print(f"  {score:.4f}: {doc}")

print("\n[FAISS]")
for doc, score in search_faiss(query):
    print(f"  {score:.4f}: {doc}")
# 문서가 적을 때는 두 방식 결과가 동일해야 정상

# 4. 대규모 데이터에서 속도 차이 확인
print("\n" + "="*50)
print("대규모 데이터 속도 비교 (가짜 벡터 10,000개)")
print("="*50)

# 실제 문장 대신 랜덤 벡터로 테스트 (임베딩 시간을 빼고 검색 속도만 비교)
np.random.seed(42)
large_vectors = np.random.random((10000, dimension)).astype('float32')
faiss.normalize_L2(large_vectors)

query_vec = np.random.random((1, dimension)).astype('float32')
faiss.normalize_L2(query_vec)

# --- 브루트포스 ---
start = time.time()
scores = np.dot(large_vectors, query_vec.T).flatten()
top_indices_bf = np.argsort(scores)[::-1][:5]
bf_time = time.time() - start
print(f"부루트포스: {bf_time*1000:.2f}ms")

# --- FAISS ---
large_index = faiss.IndexFlatIP(dimension)
large_index.add(large_vectors)

start = time.time()
faiss_scores, faiss_indices = large_index.search(query_vec, 5)
faiss_time = time.time() - start
print(f"FAISS:    {faiss_time*1000:.2f}ms")

print(f"\n결과 일치 여부 : {set(top_indices_bf) == set(faiss_indices[0])}")

# 5. 인덱스 저장 & 불러오기 (서버 재시작 시 재계산 방지)
# 실제 서비스에서 중요: 서버를 켤 때마다 모든 문서를 다시 임베딩하면 너무 느림
# 인덱스를 파일로 저장해두고 불러오면 즉시 사용 가능
faiss.write_index(index, "documents.index")
print("\n인덱스 저장 완료: documents.index")

loaded_index = faiss.read_index("documents.index")
print("불러온 인덱스 벡터 수:", loaded_index.ntotal)

# 6. TODO
print("\nTODO")
# 6-1: 다른 질문("한국 음식 알려줘")으로 두 방식 결과가 같은지 확인
query1 = "한국 음식 알려줘"

print("\n[브루트포스]")
for doc, score in search_bruteforce(query1):
    print(f"  {score:.4f}: {doc}")

print("\n[FAISS]")
for doc, score in search_faiss(query1):
    print(f"  {score:.4f}: {doc}")

# 6-2: large_vectors를 100,000개로 늘려서 속도 차이가 더 벌어지는지 확인
#      (메모리 주의: 100,000 x 768 x 4바이트 = 약 300MB)
np.random.seed(42)
large_vectors1 = np.random.random((100000, dimension)).astype('float32')
faiss.normalize_L2(large_vectors1)

query_vec1 = np.random.random((1, dimension)).astype('float32')
faiss.normalize_L2(query_vec1)

# --- 브루트포스 ---
start1 = time.time()
scores1 = np.dot(large_vectors1, query_vec1.T).flatten()
top_indices_bf1 = np.argsort(scores1)[::-1][:5]
bf_time1 = time.time() - start1
print(f"브루트포스: {bf_time1*1000:.2f}ms")

# --- FAISS ---
large_index1 = faiss.IndexFlatIP(dimension)
large_index1.add(large_vectors1)

start1 = time.time()
faiss_scores1, faiss_indices1 = large_index1.search(query_vec1, 5)
faiss_time1 = time.time() - start1
print(f"FAISS:    {faiss_time1*1000:.2f}ms")

# 6-3: index.add()로 새 문서를 추가한 뒤 검색 결과에 반영되는지 확인
#      (힌트: 새 문장을 encode -> normalize_L2 -> index.add)
# 기존 인덱스 상태 확인
print("추가 전 벡터 수:", index.ntotal)   # 6

# 새 문서 준비
new_doc = "감기는 바이러스로 인해 코와 목에 염증이 생기는 흔한 호흡기 질환이다."

# 1) 임베딩
new_vec = embed_model.encode(new_doc).astype('float32').reshape(1, -1)
# 2) 정규화 (기존 벡터들과 동일한 처리)
faiss.normalize_L2(new_vec)
# 3) 기존 인덱스에 추가
index.add(new_vec)

print("추가 후 벡터 수:", index.ntotal)   # 7

# documents 리스트에도 추가 (인덱스 번호와 문서를 매칭해야 하므로)
documents.append(new_doc)

# 검색해서 새 문서가 잡히는지 확인
print(search_faiss("감기 걸렸을 때 어떡해?"))