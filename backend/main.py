"""
FastAPI Main Application
AI Text Detector Backend API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    AnalyzeRequest, AnalyzeResponse,
    CheckpointLoadRequest, CheckpointLoadResponse,
    CheckpointListResponse, ParagraphInfo, TopParagraph,
    LimeToken, LimeResult, DeletionTest
)
from .model_loader import model_loader
from .inference import inference_service
from .lime_analyzer import lime_analyzer


# Create FastAPI app
app = FastAPI(
    title="AI Text Detector API",
    description="Hierarchical Explanation API for AI-generated text detection",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    model_loader.load_checkpoint("default")
    print("✅ Model loaded successfully")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "AI Text Detector API"}


@app.get("/checkpoints", response_model=CheckpointListResponse)
async def list_checkpoints():
    """
    Get list of available model checkpoints
    """
    checkpoints = model_loader.list_checkpoints()
    return CheckpointListResponse(checkpoints=checkpoints)


@app.post("/checkpoints/load", response_model=CheckpointLoadResponse)
async def load_checkpoint(request: CheckpointLoadRequest):
    """
    Load a specific model checkpoint
    """
    success = model_loader.load_checkpoint(request.checkpoint_name)
    
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to load checkpoint: {request.checkpoint_name}")
    
    return CheckpointLoadResponse(
        status="loaded",
        model_name=request.checkpoint_name
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """
    Full document analysis:
    1. Split text into paragraphs
    2. Calculate AI probability for each paragraph
    3. Calculate Leave-One-Out importance
    4. Run LIME on the top paragraph
    5. Run deletion test for reliability verification
    """
    text = request.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Step 1-3: Document analysis
    doc_analysis = inference_service.analyze_document(text)
    
    # Step 4-5: LIME + Deletion test on top paragraph
    top_para_text = doc_analysis['top_paragraph']['text']
    lime_analysis = lime_analyzer.full_analysis(top_para_text)
    
    # Build response
    paragraphs = [
        ParagraphInfo(
            index=p['index'],
            text=p['text'],
            ai_prob=p['ai_prob'],
            importance=p['importance']
        )
        for p in doc_analysis['paragraphs']
    ]
    
    top_paragraph = TopParagraph(
        index=doc_analysis['top_paragraph']['index'],
        text=doc_analysis['top_paragraph']['text'],
        ai_prob=doc_analysis['top_paragraph']['ai_prob']
    )
    
    lime_tokens = [
        LimeToken(word=t['word'], score=t['score'])
        for t in lime_analysis['lime_result']['tokens']
    ]
    
    deletion_test = DeletionTest(
        original_prob=lime_analysis['deletion_test']['original_prob'],
        modified_prob=lime_analysis['deletion_test']['modified_prob'],
        drop=lime_analysis['deletion_test']['drop'],
        reliability=lime_analysis['deletion_test']['reliability'],
        removed_tokens=lime_analysis['deletion_test']['removed_tokens']
    )
    
    return AnalyzeResponse(
        prediction=doc_analysis['prediction'],
        confidence=doc_analysis['confidence'],
        paragraphs=paragraphs,
        top_paragraph=top_paragraph,
        lime_result=LimeResult(tokens=lime_tokens),
        deletion_test=deletion_test
    )


# For running with: uvicorn backend.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
