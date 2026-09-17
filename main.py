import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 앱 초기화 & 모델 로딩
# 모델 로딩은 서버 시작 시 한 번만 실행됨
# (요청마다 로딩 시 매번 수십 초가 걸려 서비스 X)
app = FastAPI(title="RAG API", description="검색 기반 질의응답 API")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

print("임베딩 모델 로딩 중...")
embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

print("생성 모델 로딩 중... (최초 실행 시 다운로드로 몇 분 소요)")

gen_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(
    gen_model_name,
    torch_dtype=torch.float32,  # CPU는 float16 지원이 불안정하므로 float32 사용
)
gen_model.to(device)
gen_model.eval()
print("모델 로딩 완료")

# 2. 지식 베이스
documents = [
    "타이타닉은 1912년 4월 15일 빙산과 충돌해 침몰한 영국의 여객선이다.",
    "파이썬은 1991년 귀도 반 로섬이 개발한 프로그래밍 언어이다.",
    "BERT는 구글이 2018년에 발표한 자연어처리 모델이다.",
    "김치는 발효 채소를 이용한 한국의 전통 음식이다.",
    "RAG는 검색과 생성을 결합한 자연어처리 기법이다.",
    "딥러닝은 인공신경망을 여러 층으로 쌓아 학습하는 머신러닝의 한 분야이다.",
]
doc_embeddings = embed_model.encode(documents)

# 3. 요청/응답 형식 정의 (Pydantic)
class QueryRequest(BaseModel):
    question: str
    top_k: int = 2

class QueryResponse(BaseModel):
    question: str
    retrieved_docs: list[str]
    answer: str

class SearchResponse(BaseModel):
    question: str
    results: list[dict]

# 4. RAG 로직
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

def search_with_scores(query, top_k=2):
    """유사도 점수까지 함께 반환"""
    query_vec = embed_model.encode(query)
    scores = [cosine_sim(query_vec, doc_vec) for doc_vec in doc_embeddings]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"document": documents[i], "score": float(scores[i])} for i in top_indices]

def generate_answer(query, context):
    prompt = f"""다음 참고 자료를 바탕으로 질문에 답하세요. 참고 자료에 없는 내용은 답하지 마세요.

참고 자료:
{context}

질문: {query}
답변:"""
    messages = [{"role": "user", "content": prompt}]
    text = gen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = gen_tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
    return gen_tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    ).strip()

# 5. API 엔드포인트
@app.get("/")
def root():
    """서버 상태 확인 (헬스 체크)"""
    return {"status": "running", "device": str(device)}

@app.get("/documents")
def list_documents():
    """지식 베이스에 등록된 문서 목록 조회"""
    return {"count": len(documents), "documents": documents}

@app.post("/search", response_model=SearchResponse)
def search_only(request: QueryRequest):
    """검색만 수행 (생성 없이 빠르게 확인용)"""
    results = search_with_scores(request.question, top_k=request.top_k)
    return SearchResponse(question=request.question, results=results)

@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    """RAG 전체 파이프라인: 검색 + 답변 생성"""
    results = search_with_scores(request.question, top_k=request.top_k)
    retrieved = [r["document"] for r in results]
    context = "\n".join(retrieved)
    answer = generate_answer(request.question, context)

    return QueryResponse(
        question=request.question,
        retrieved_docs=retrieved,
        answer=answer
    )