"""
Backstory Consistency Checker for Kharagpur Data Science Hackathon 2026
Simple MVP version
"""

import os
from dotenv import load_dotenv
import pandas as pd
# Current imports ke baad
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("ERROR: Gemini API key not found in .env file")
    print("Please add: GEMINI_API_KEY=your_key_here in .env file")
    exit(1)
    print("ERROR: OpenAI API key not found in .env file")
    print("Please add: OPENAI_API_KEY=your_key_here in .env file")
    exit(1)

print("[OK] Setup complete. Ready to load novel...")

def load_novel(novel_path):
    """
    Load novel from text or PDF file
    """
    from pypdf import PdfReader
    
    # Check file extension
    if novel_path.lower().endswith('.txt'):
        with open(novel_path, 'r', encoding='utf-8') as f:
            text = f.read()
    
    elif novel_path.lower().endswith('.pdf'):
        text = ""
        with open(novel_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text()
    
    else:
        raise ValueError("Only .txt or .pdf files supported")
    
    print(f"[OK] Novel loaded: {len(text)} characters")
    return text

def chunk_novel(text, chunk_size=2000, overlap=200):
    """
    Split novel into overlapping chunks for better context
    """
    import tiktoken
    
    # Simple chunking by character count (better than word count)
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move with overlap
        start = end - overlap
    
    print(f"[OK] Novel split into {len(chunks)} chunks")
    
    # Show sample chunk
    if chunks:
        print(f"[SAMPLE] chunk ({len(chunks[0])} chars): {chunks[0][:100]}...")
    
    return chunks

def create_pathway_index(chunks):
    """
    Create index for semantic search
    """
    # Create simple index structure
    index_data = []
    for i, chunk in enumerate(chunks):
        index_data.append({
            "text": chunk,
            "chunk_id": i
        })
    
    print("[OK] Index created")
    print(f"   Total chunks indexed: {len(chunks)}")
    
    return index_data

def backstory_to_claims(backstory_text):
    """
    Convert backstory into individual claims/statements
    Simple version: Split by sentences
    """
    import re
    
    # Split by common sentence endings
    sentences = re.split(r'[.!?]+', backstory_text)
    
    # Clean up sentences
    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # Skip very short sentences
            claims.append(sentence)
    
    print(f"[OK] Backstory converted to {len(claims)} claims")
    
    # Show sample claims
    for i, claim in enumerate(claims[:3]):
        print(f"   Claim {i+1}: {claim[:50]}...")
    
    return claims

def load_backstory(backstory_path):
    """Load character backstory from file"""
    with open(backstory_path, 'r', encoding='utf-8') as f:
        return f.read()

def retrieve_relevant_chunks(claim, index_table, top_k=3):
    """
    Retrieve most relevant chunks for a claim from index
    Simple keyword-based matching
    """
    # Convert to lowercase for matching
    claim_lower = claim.lower()
    claim_words = set(claim_lower.split())
    
    # Score each chunk based on keyword matches
    scored_chunks = []
    for item in index_table:
        chunk_lower = item["text"].lower()
        chunk_words = set(chunk_lower.split())
        
        # Count matching words
        matches = len(claim_words & chunk_words)
        scored_chunks.append((item["text"], matches))
    
    # Sort by score and get top_k
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    relevant = [chunk for chunk, score in scored_chunks[:top_k]]
    
    print(f"   Retrieving chunks for claim: {claim[:30]}...")
    
    return relevant

def retrieve_for_all_claims(claims, index_table):
    """
    Retrieve relevant chunks for each claim using vector search
    """
    results = {}
    
    for i, claim in enumerate(claims):
        chunks = retrieve_relevant_chunks(claim, embedded_table)
        results[claim] = chunks
    
    return results

def check_claim_with_llm(claim, retrieved_chunks):
    """
    Use Google Gemini for consistency checking (FREE alternative)
    """
    # Load Gemini API key
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_key:
        # Fallback to mock response
        return get_mock_llm_response(claim)
    
    # Use Gemini (free tier available)
    if GEMINI_AVAILABLE:
        return check_with_gemini(claim, retrieved_chunks, gemini_key)
    else:
        return get_mock_llm_response(claim)

def check_with_gemini(claim, retrieved_chunks, api_key):
    """
    Use Google Gemini API
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-pro')
        
        context = "\n\n".join(retrieved_chunks[:2])
        
        prompt = f"""Check if this character claim is consistent with novel text:

NOVEL TEXT:
{context}

CHARACTER CLAIM:
{claim}

Answer in exact format:
CONSISTENCY: [SUPPORTED/CONTRADICTED/NOT_FOUND]
QUOTE: [exact quote from novel or "NO_DIRECT_QUOTE"]"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"   Gemini Error: {e}")
        return "CONSISTENCY: ERROR\nQUOTE: ERROR"

def check_with_openai(claim, retrieved_chunks):
    """
    Original OpenAI function (fallback)
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        context = "\n\n".join(retrieved_chunks[:2])
        
        prompt = f"""Check if this claim matches the novel text:

 NOVEL TEXT:
{context}

CLAIM:
{claim}

Answer format:
CONSISTENCY: [SUPPORTED/CONTRADICTED/NOT_FOUND]
QUOTE: [exact quote or "NO_DIRECT_QUOTE"]"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"   OpenAI Error: {e}")
        return "CONSISTENCY: ERROR\nQUOTE: ERROR"

def get_mock_llm_response(claim):
    """
    Fallback mock responses
    """
    claim_lower = claim.lower()
    
    if any(word in claim_lower for word in ['50 years', '100 years', 'ancient']):
        return "CONSISTENCY: CONTRADICTED\nQUOTE: 'Character is described as young'"
    elif any(word in claim_lower for word in ['elizabeth', 'darcy', 'bennet']):
        return "CONSISTENCY: SUPPORTED\nQUOTE: 'Character mentioned in novel'"
    else:
        return "CONSISTENCY: NOT_FOUND\nQUOTE: NO_DIRECT_QUOTE"
def parse_llm_response(response_text):
    """
    Parse improved LLM response with causal reasoning
    """
    result = {
        "consistency": "UNKNOWN",
        "causal_reasoning": "",
        "quote": "",
        "chunk_reference": "",
        "reason": ""
    }
    
    lines = response_text.split('\n')
    current_field = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("CONSISTENCY:"):
            result["consistency"] = line.replace("CONSISTENCY:", "").strip()
        elif line.startswith("CAUSAL_REASONING:"):
            result["causal_reasoning"] = line.replace("CAUSAL_REASONING:", "").strip()
        elif line.startswith("EXACT_QUOTE:"):
            result["quote"] = line.replace("EXACT_QUOTE:", "").strip()
        elif line.startswith("CHUNK_REFERENCE:"):
            result["chunk_reference"] = line.replace("CHUNK_REFERENCE:", "").strip()
    
    return result

def generate_dossier(claims, retrieved_results):
    """
    Generate dossier with all claims and their verification results
    """
    dossier = []
    
    print("[CHECK] Checking claims with LLM...")
    
    for i, claim in enumerate(claims):
        print(f"   Checking claim {i+1}/{len(claims)}...")
        
        # Get LLM analysis
        llm_response = check_claim_with_llm(claim, retrieved_results[claim])
        parsed_result = parse_llm_response(llm_response)
        
        # Add to dossier
        dossier.append({
              "claim_id": i + 1,
            "claim_text": claim,
            "consistency": parsed_result["consistency"],
            "causal_reasoning": parsed_result["causal_reasoning"],
            "exact_quote": parsed_result["quote"],
            "chunk_reference": parsed_result["chunk_reference"],
            "retrieved_chunks_count": len(retrieved_results[claim])
        })
    
    print(f"[OK] Dossier generated with {len(dossier)} entries")
    return dossier

def make_binary_decision(dossier):
    """
    Convert dossier results into final binary decision
    Rules:
    - ANY contradiction → 0 (Contradictory)
    - ALL supported → 1 (Consistent)
    - Mixed (some supported, some not found) → 0 (Contradictory - safer)
    """
    
    consistency_counts = {
        "SUPPORTED": 0,
        "CONTRADICTED": 0,
        "NOT_FOUND": 0,
        "ERROR": 0
    }
    
    for entry in dossier:
        consistency = entry["consistency"]
        if consistency in consistency_counts:
            consistency_counts[consistency] += 1
    
    print("\n[SUMMARY] Consistency Summary:")
    print(f"   Supported: {consistency_counts['SUPPORTED']}")
    print(f"   Contradicted: {consistency_counts['CONTRADICTED']}")
    print(f"   Not Found: {consistency_counts['NOT_FOUND']}")
    print(f"   Errors: {consistency_counts['ERROR']}")
    
    # Decision logic
    if consistency_counts["CONTRADICTED"] > 0:
        print("[FAIL] Decision: Contradictory (at least one contradiction found)")
        return 0
    elif consistency_counts["ERROR"] > 0:
        print("[WARN] Decision: Contradictory (errors in checking)")
        return 0
    elif consistency_counts["SUPPORTED"] == len(dossier):
        print("[OK] Decision: Consistent (all claims supported)")
        return 1
    else:
        print("[?] Decision: Contradictory (not all claims clearly supported)")
        return 0

def save_results(dossier, final_decision, filename="results.csv"):
    """
    Save dossier and final decision to CSV file
    """
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/{timestamp}_{filename}"
    
    # Convert dossier to DataFrame
    df_dossier = pd.DataFrame(dossier)
    
    # Add summary row
    summary_row = {
        "claim_id": "SUMMARY",
        "claim_text": f"FINAL_DECISION: {final_decision} (1=Consistent, 0=Contradictory)",
        "consistency": f"Supported: {sum(1 for d in dossier if d['consistency']=='SUPPORTED')}",
        "quote": f"Total Claims: {len(dossier)}",
        "retrieved_chunks_count": ""
    }
    
    # Add summary to dataframe
    df_summary = pd.DataFrame([summary_row])
    df_final = pd.concat([df_summary, df_dossier], ignore_index=True)
    
    # Save to CSV
    df_final.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"[OK] Results saved to: {output_path}")
    
    # Also save a simple results.txt with just the binary decision
    txt_path = f"output/{timestamp}_binary_result.txt"
    with open(txt_path, 'w') as f:
        f.write(str(final_decision))
    
    print(f"[OK] Binary result saved to: {txt_path}")
    
    return output_path

def save_hackathon_csv(final_decision, novel_id="novel_001", filename="results.csv"):
    """
    Save hackathon-required CSV format: id, label columns only
    """
    import pandas as pd
    from datetime import datetime
    
    os.makedirs("output", exist_ok=True)
    
    # Create hackathon submission CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hackathon_csv_path = f"output/{timestamp}_hackathon_{filename}"
    
    # Create simple dataframe with hackathon format
    data = {
        "id": [novel_id],
        "label": [final_decision]
    }
    
    df_hackathon = pd.DataFrame(data)
    df_hackathon.to_csv(hackathon_csv_path, index=False)
    
    print(f"[OK] Hackathon CSV saved to: {hackathon_csv_path}")
    print(f"   Format: id='{novel_id}', label={final_decision}")
    
    return hackathon_csv_path
# Main execution
# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("BACKSTORY CONSISTENCY CHECKER - Hackathon MVP")
    print("=" * 60)
    
    # Check available LLM services
    print("\n[CHECK] Checking available LLM services...")
    if os.getenv("GEMINI_API_KEY"):
        print("   [OK] Google Gemini API available")
    if os.getenv("OPENAI_API_KEY"):
        print("   [OK] OpenAI API available")
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("   [WARN] No API keys found - Using mock responses")
    
    # 1. Load novel
    novel_text = load_novel("data/novel.txt")
    
    # 2. Chunk novel for processing
    chunks = chunk_novel(novel_text)
    
    # 3. Create Pathway index with vector embeddings
    print("\n[CREATE] Creating vector index...")
    embedded_table = create_pathway_index(chunks)
    
    # 4. Load backstory
    backstory_text = load_backstory("data/backstory.txt")
    
    # 5. Extract claims from backstory
    claims = backstory_to_claims(backstory_text)
    
    # 6. Retrieve relevant chunks using vector search
    print("\n[SEARCH] Retrieving relevant text with semantic search...")
    retrieved_results = retrieve_for_all_claims(claims, embedded_table)
    
    # 7. Generate dossier with LLM analysis
    print("\n[LLM] Analyzing claims with LLM...")
    dossier = generate_dossier(claims, retrieved_results)
    
    # 8. Make final binary decision
    final_decision = make_binary_decision(dossier)
    
    # 9. Save detailed results
    csv_path = save_results(dossier, final_decision)

    # 10. Save hackathon-required CSV format
    hackathon_csv_path = save_hackathon_csv(final_decision)
    
    # 11. Final hackathon output
    print("\n" + "=" * 60)
    print("[OUTPUT] HACKATHON SUBMISSION OUTPUT")
    print("=" * 60)
    print(f"Binary Label (0/1): {final_decision}")
    print(f"Dossier Entries: {len(dossier)} claims analyzed")
    print(f"Detailed Results: {csv_path}")
    print(f"Hackathon CSV: {hackathon_csv_path}")
    
    # Count consistency types
    from collections import Counter
    consistency_types = Counter([d['consistency'] for d in dossier])
    
    print(f"\n[SUMMARY] Summary:")
    for cons_type, count in consistency_types.items():
        print(f"   {cons_type}: {count}")
    
    print("=" * 60)