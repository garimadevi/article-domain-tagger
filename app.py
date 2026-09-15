"""FastAPI backend for the Article Domain Tagger.

Loads the trained student model and serves predictions via REST API.

Run:
    uvicorn app:app --reload --port 8000

Endpoints:
    POST /predict       — classify article text, returns top-3 labels
    POST /fetch-url     — fetch and extract article from URL, then classify
    GET  /health        — model status check
"""

from pathlib import Path
from contextlib import asynccontextmanager
import re
import asyncio

import torch
import requests
from readability import Document
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = Path(__file__).resolve().parent / "generated" / "deployed_student"
MAX_LEN = 96
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = None
tokenizer = None

# URL fetching config
FETCH_TIMEOUT = 15
MAX_CONTENT_LENGTH = 500000
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# More realistic headers to avoid blocking
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# Playwright config
PLAYWRIGHT_TIMEOUT = 45000
PLAYWRIGHT_WAIT_UNTIL = "domcontentloaded"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print(f"[app] loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[app] model loaded ({n_params:,} params, {DEVICE})")
    yield
    print("[app] shutting down")


app = FastAPI(
    title="Article Domain Tagger API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Article text to classify")


class FetchUrlRequest(BaseModel):
    url: HttpUrl = Field(..., description="Article URL to fetch and classify")


class LabelScore(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    top3: list[LabelScore]
    all_scores: dict[str, float]


class FetchUrlResponse(BaseModel):
    url: str
    title: str
    extracted_text: str
    prediction: str
    confidence: float
    top3: list[LabelScore]
    all_scores: dict[str, float]


def extract_article_from_html(html: str, url: str) -> tuple[str, str]:
    """Extract title and main text from HTML using readability."""
    try:
        doc = Document(html)
        title = doc.title() or "Untitled"
        content_html = doc.summary()
        
        # Clean up the extracted text
        text = re.sub(r'<[^>]+>', ' ', content_html)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return title, text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract article: {str(e)}")


def fetch_and_extract_sync(url: str) -> tuple[str, str]:
    """Fetch URL and extract article title and text (sync version with requests)."""
    
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=FETCH_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
    except requests.Timeout:
        raise HTTPException(status_code=408, detail="Request timed out")
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
    
    # Check content type
    content_type = resp.headers.get("content-type", "").lower()
    if "text/html" not in content_type and "application/xml" not in content_type and "application/rss" not in content_type:
        # Try to parse anyway
        pass
    
    # Limit content size - need to slice the content before decoding
    content = resp.content
    if len(content) > MAX_CONTENT_LENGTH:
        content = content[:MAX_CONTENT_LENGTH]
    
    # Detect encoding
    encoding = resp.encoding or resp.apparent_encoding or "utf-8"
    html = content.decode(encoding, errors="replace")
    
    return extract_article_from_html(html, url)


async def fetch_and_extract_with_playwright(url: str) -> tuple[str, str]:
    """Fetch URL using Playwright (headless browser) and extract article."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise HTTPException(status_code=500, detail="Playwright not installed. Run: pip install playwright && playwright install chromium")
    
    print(f"[playwright] Starting async_playwright")
    async with async_playwright() as p:
        print(f"[playwright] Launching browser")
        browser = await p.chromium.launch(headless=True)
        print(f"[playwright] Browser launched")
        try:
            context = await browser.new_context(
                user_agent=USER_AGENT,
                viewport={"width": 1280, "height": 720},
            )
            page = await context.new_page()
            
            # Block unnecessary resources to speed up
            await page.route("**/*", lambda route: route.abort() 
                if route.request.resource_type in ["image", "font", "media", "stylesheet"] 
                else route.continue_())
            
            print(f"[playwright] Navigating to: {url}")
            await page.goto(url, wait_until=PLAYWRIGHT_WAIT_UNTIL, timeout=PLAYWRIGHT_TIMEOUT)
            
            # Wait a bit for dynamic content
            await page.wait_for_timeout(2000)
            
            # Get the rendered HTML
            html = await page.content()
            print(f"[playwright] Got HTML length: {len(html)}")
            
            return extract_article_from_html(html, url)
        finally:
            print(f"[playwright] Closing browser")
            await browser.close()


async def fetch_with_fallback(url: str) -> tuple[str, str]:
    """Try requests first, fallback to Playwright if needed."""
    # Try fast path first
    try:
        title, text = fetch_and_extract_sync(url)
        if text and len(text.strip()) >= 100:
            print(f"[fetch] Success with requests: {len(text)} chars")
            return title, text
        print(f"[fetch] Requests returned insufficient text ({len(text)} chars), trying Playwright...")
    except HTTPException as e:
        if e.status_code in (401, 403, 404, 408, 500):
            print(f"[fetch] Requests failed with {e.status_code}, trying Playwright...")
        else:
            raise
    except Exception as e:
        print(f"[fetch] Requests failed: {e}, trying Playwright...")
    
    # Fallback to Playwright
    try:
        title, text = await fetch_and_extract_with_playwright(url)
        if text and len(text.strip()) >= 50:
            print(f"[fetch] Success with Playwright: {len(text)} chars")
            return title, text
        raise HTTPException(status_code=400, detail="Could not extract sufficient text from the article (even with browser rendering)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Browser rendering failed: {str(e)}")


def run_prediction(text: str):
    """Run model prediction on text."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    scores = probs.cpu().numpy()
    id2label = model.config.id2label

    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    top3 = [LabelScore(label=id2label[i], confidence=round(float(s), 4)) for i, s in ranked[:3]]

    all_scores = {id2label[i]: round(float(s), 4) for i, s in enumerate(scores)}

    best_idx, best_score = ranked[0]
    return PredictResponse(
        prediction=id2label[best_idx],
        confidence=round(float(best_score), 4),
        top3=top3,
        all_scores=all_scores,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": DEVICE,
        "num_labels": model.config.num_labels if model else 0,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return run_prediction(req.text)


@app.post("/fetch-url", response_model=FetchUrlResponse)
async def fetch_url(req: FetchUrlRequest):
    url = str(req.url)
    print(f"[fetch-url] Fetching: {url}")
    try:
        title, extracted_text = await fetch_with_fallback(url)
        print(f"[fetch-url] Extracted: title='{title}', text_len={len(extracted_text)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[fetch-url] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    
    if not extracted_text or len(extracted_text.strip()) < 50:
        print(f"[fetch-url] Insufficient text extracted")
        raise HTTPException(status_code=400, detail="Could not extract sufficient text from the article")
    
    # Truncate if too long
    if len(extracted_text) > 5000:
        extracted_text = extracted_text[:5000]
    
    try:
        prediction = run_prediction(extracted_text)
        print(f"[fetch-url] Prediction: {prediction.prediction} ({prediction.confidence})")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[fetch-url] Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
    
    return FetchUrlResponse(
        url=url,
        title=title,
        extracted_text=extracted_text,
        prediction=prediction.prediction,
        confidence=prediction.confidence,
        top3=prediction.top3,
        all_scores=prediction.all_scores,
    )