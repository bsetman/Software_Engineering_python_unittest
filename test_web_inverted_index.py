import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from web_inverted_index import app, Base, InvertedIndex, SessionLocal, elias_gamma_encode, elias_gamma_decode

client = TestClient(app)


TEST_DB_URL = "sqlite:///:memory:"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(bind=engine)


Base.metadata.create_all(bind=engine)
app.dependency_overrides[SessionLocal] = TestingSessionLocal

@pytest.fixture(autouse=True)
def clean_db():

    session = TestingSessionLocal()
    session.query(InvertedIndex).delete()
    session.commit()
    session.close()

def test_index_pages_success(monkeypatch):
    test_html = "<html><body>Hello FastAPI World!</body></html>"
    monkeypatch.setattr("requests.get", lambda url: type("obj", (object,), {"status_code": 200, "text": test_html}))

    response = client.post("/index_pages", json={"urls": ["http://example.com"]})
    assert response.status_code == 200
    assert "indexed_terms" in response.json()

def test_index_pages_failure(monkeypatch):
    monkeypatch.setattr("requests.get", lambda url: type("obj", (object,), {"status_code": 404, "text": ""}))

    response = client.post("/index_pages", json={"urls": ["http://nonexistent.com"]})
    assert response.status_code == 404

def test_search_existing_term():
    # Вручную вставьте термин fastapi -> doc_id 1
    session = TestingSessionLocal()
    bitstring = "01001"  # Elias Gamma(1) = "1"
    session.add(InvertedIndex(term="fastapi", postings=bitstring.encode("utf-8")))
    session.commit()
    session.close()

    response = client.get("/search", params={"term": "fastapi"})
    assert response.status_code == 200
    assert response.json() == {"term": "fastapi", "postings": [1]}

def test_search_nonexistent_term():
    response = client.get("/search", params={"term": "nonexistentterm"})
    assert response.status_code == 404

def test_elias_gamma_compression():


    original = [1, 3, 7]
    diffs = [original[0]] + [original[i] - original[i - 1] for i in range(1, len(original))]
    bits = "".join(elias_gamma_encode(d) for d in diffs)
    decoded = elias_gamma_decode(bits)


    result = []
    acc = 0
    for d in decoded:
        acc += d
        result.append(acc)

    assert result == original
