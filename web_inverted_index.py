import os
import re
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ==========================================
# Configuration: HTTP and Database Ports
# ==========================================
# HTTP_PORT: the port on which this service listens (default: 8000)
# DB_PORT: the port for the PostgreSQL database (default: 5432)
# Other DB settings: DB_HOST, DB_USER, DB_PASSWORD, DB_NAME

HTTP_PORT = int(os.getenv("HTTP_PORT", 8000))
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_USER = os.getenv("DB_USER", "youruser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "yourpassword")
DB_NAME = os.getenv("DB_NAME", "inverted_index_db")

DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# =============================
# Database Setup (SQLAlchemy)
# =============================
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class InvertedIndex(Base):
    __tablename__ = "inverted_index"
    term = Column(String, primary_key=True, index=True)
    # Store compressed index bitstring as bytes
    postings = Column(LargeBinary)

# Create table if not exists
Base.metadata.create_all(bind=engine)

# =============================
# FastAPI App Initialization
# =============================
app = FastAPI(
    title="Web Inverted Index Service",
    description="Index web pages and store compressed inverted index in PostgreSQL",
)

# Request model for bulk page indexing
class Pages(BaseModel):
    urls: list[str]

# ======================================
# Inverted-Index & Compression Functions
# ======================================

def build_inverted_index(documents: dict[int, str]) -> dict[str, list[int]]:
    inverted_index: dict[str, list[int]] = {}
    for doc_id, text in documents.items():
        words = re.findall(r"\w+", text.lower())
        for term in words:
            postings = inverted_index.setdefault(term, [])
            if not postings or postings[-1] != doc_id:
                postings.append(doc_id)
    for term in inverted_index:
        inverted_index[term] = sorted(set(inverted_index[term]))
    return inverted_index


def elias_gamma_encode(n: int) -> str:
    if n <= 0:
        raise ValueError("Elias gamma only for n>=1")
    binary = bin(n)[2:]
    N = len(binary) - 1
    return "0" * N + binary


def compress_index_gamma(inverted_index: dict[str, list[int]]) -> dict[str, str]:
    compressed: dict[str, str] = {}
    for term, postings in inverted_index.items():
        prev = 0
        bits = ""
        for doc_id in postings:
            diff = doc_id - prev
            bits += elias_gamma_encode(diff)
            prev = doc_id
        compressed[term] = bits
    return compressed


def elias_gamma_decode(bitstring: str) -> list[int]:
    results: list[int] = []
    i, n = 0, len(bitstring)
    while i < n:
        zeros = 0
        while i < n and bitstring[i] == "0":
            zeros += 1
            i += 1
        if i >= n: break
        length = zeros + 1
        segment = bitstring[i : i + length]
        i += length
        results.append(int(segment, 2))
    return results


def decompress_index_gamma(compressed: dict[str, str]) -> dict[str, list[int]]:
    inverted: dict[str, list[int]] = {}
    for term, bits in compressed.items():
        diffs = elias_gamma_decode(bits)
        postings, cum = [], 0
        for d in diffs:
            cum += d
            postings.append(cum)
        inverted[term] = postings
    return inverted

# =============================
# API Endpoints
# =============================

@app.post("/index_pages")
async def index_pages(pages: Pages):
    """
    Fetch each URL, extract text, build and compress inverted index,
    and store into PostgreSQL.
    """
    session = SessionLocal()
    try:
        documents: dict[int, str] = {}
        for idx, url in enumerate(pages.urls, start=1):
            resp = requests.get(url)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Failed to fetch {url}",
                )
            # Simple text extraction
            text = " ".join(re.findall(r"\w+", resp.text.lower()))
            documents[idx] = text

        inverted = build_inverted_index(documents)
        compressed = compress_index_gamma(inverted)

        # Upsert into DB
        for term, bits in compressed.items():
            # Store as UTF-8 bytes
            entry = InvertedIndex(term=term, postings=bits.encode("utf-8"))
            session.merge(entry)
        session.commit()
        return {"indexed_terms": len(compressed)}
    finally:
        session.close()

@app.get("/search")
async def search_term(term: str):
    """
    Retrieve compressed postings for a term, decompress, and return list of doc IDs.
    """
    session = SessionLocal()
    try:
        record = session.query(InvertedIndex).filter_by(term=term).first()
        if not record:
            raise HTTPException(status_code=404, detail="Term not found")
        bitstring = record.postings.decode("utf-8")
        postings = decompress_index_gamma({term: bitstring})[term]
        return {"term": term, "postings": postings}
    finally:
        session.close()

# =============================
# Server Startup
# =============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "web_inverted_index:app",
        host="0.0.0.0",
        port=HTTP_PORT,
        reload=True,
    )
