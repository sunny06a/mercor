"""
Mercor Candidate Search and Re-ranking System

"""

import turbopuffer as tpuf
from sentence_transformers import CrossEncoder
import voyageai
import requests
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
TPUF_API_KEY = "tpuf_wTbagsVtzNmVfzDm48lNeszzJdTaCOUF"
VOYAGE_API_KEY = "pa-vNEmoJfc5evP_SSvpxIAj3uFzs9dfppEZkpx-3kOFZy"
EVALUATION_EMAIL = "sunnytomar3786@gmail.com"
COLLECTION_NAME = "search-test-v4"
TURBOPUFFER_REGION = "aws-us-west-2"

# Model configuration
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
VOYAGE_MODEL = "voyage-3"
INITIAL_RETRIEVAL_SIZE = 50
FINAL_RESULTS_SIZE = 10


def initialize_clients():
    """Initialize Turbopuffer, Voyage AI, and cross-encoder clients."""
    logger.info("Initializing clients and loading models")
    
    tpuf_client = tpuf.Turbopuffer(
        api_key=TPUF_API_KEY,
        region=TURBOPUFFER_REGION
    )
    
    vo_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    reranker = CrossEncoder(CROSS_ENCODER_MODEL)
    
    logger.info("Clients initialized successfully")
    return tpuf_client, vo_client, reranker


def get_embedding(text, vo_client):
    """
    Generate vector embedding for input text using Voyage AI.
    
    Args:
        text (str): Input text to embed
        vo_client: Voyage AI client instance
        
    Returns:
        list: 1024-dimensional embedding vector
    """
    result = vo_client.embed([text], model=VOYAGE_MODEL)
    return result.embeddings[0]


def extract_rerank_summary(row):
    """
    Extract rerankSummary field from Turbopuffer row object.
    
    Args:
        row: Turbopuffer row object
        
    Returns:
        str: Candidate summary text for re-ranking
    """
    if hasattr(row, 'attributes') and row.attributes:
        return row.attributes.get('rerankSummary', '')
    elif hasattr(row, 'rerankSummary'):
        return row.rerankSummary
    elif isinstance(row, dict):
        return row.get('rerankSummary', '')
    else:
        for attr in ['rerank_summary', 'summary', 'description', 'bio']:
            if hasattr(row, attr):
                return getattr(row, attr)
        return ''


def extract_row_id(row):
    """
    Extract unique identifier from Turbopuffer row object.
    
    Args:
        row: Turbopuffer row object
        
    Returns:
        str: Candidate ID
    """
    if hasattr(row, 'id'):
        return row.id
    elif hasattr(row, '_id'):
        return row._id
    elif isinstance(row, dict):
        return row.get('id') or row.get('_id')
    else:
        return str(row)


def search_candidates(query_text, tpuf_client, vo_client, top_k=INITIAL_RETRIEVAL_SIZE):
    """
    Perform vector-based semantic search to retrieve initial candidate set.
    
    Args:
        query_text (str): Natural language search query
        tpuf_client: Turbopuffer client instance
        vo_client: Voyage AI client instance
        top_k (int): Number of candidates to retrieve
        
    Returns:
        list: Retrieved candidate rows
    """
    logger.info(f"Executing vector search: {query_text[:60]}...")
    
    embedding = get_embedding(query_text, vo_client)
    
    namespace = tpuf_client.namespace(COLLECTION_NAME)
    results = namespace.query(
        rank_by=("vector", "ANN", embedding),
        top_k=top_k,
        include_attributes=True
    )
    
    logger.info(f"Retrieved {len(results.rows)} candidates")
    return results.rows


def rerank_candidates(query_text, candidates, reranker, top_n=FINAL_RESULTS_SIZE):
    """
    Re-rank candidates using cross-encoder for improved relevance scoring.
    
    Args:
        query_text (str): Original search query
        candidates (list): Initial candidate set from vector search
        reranker: Cross-encoder model instance
        top_n (int): Number of top candidates to return
        
    Returns:
        list: Top-n re-ranked candidates
    """
    logger.info(f"Re-ranking candidates to top {top_n}")
    
    pairs = []
    valid_candidates = []
    
    for candidate in candidates:
        summary = extract_rerank_summary(candidate)
        if summary:
            pairs.append([query_text, summary])
            valid_candidates.append(candidate)
    
    if not pairs:
        logger.warning("No summaries found for re-ranking, returning initial results")
        return candidates[:top_n]
    
    scores = reranker.predict(pairs)
    ranked = sorted(zip(valid_candidates, scores), key=lambda x: x[1], reverse=True)
    
    return [candidate for candidate, score in ranked[:top_n]]


def submit_evaluation(config_path, object_ids):
    """
    Submit candidate rankings to evaluation API endpoint.
    
    Args:
        config_path (str): YAML configuration file name
        object_ids (list): Ordered list of candidate IDs
        
    Returns:
        dict: Evaluation response containing scores
    """
    url = "https://mercor-dev--search-eng-interview.modal.run/evaluate"
    headers = {
        "Content-Type": "application/json",
        "Authorization": EVALUATION_EMAIL
    }
    payload = {
        "config_path": config_path,
        "object_ids": object_ids
    }
    
    response = requests.post(url, json=payload, headers=headers)
    return response.json()


def define_queries():
    """
    Define search queries and corresponding configuration files.
    
    Returns:
        list: Tuples of (config_file, query_text)
    """
    return [
        ("tax_lawyer.yml", 
         "Seasoned attorney with JD from top U.S. law school and over 3 years of legal practice, specializing in corporate tax structuring and compliance, IRS audits, federal tax code"),
        
        ("junior_corporate_lawyer.yml", 
         "Corporate lawyer with 2-4 years experience at top-tier international law firm, M&A support, cross-border contract negotiations, reputed law school USA Europe Canada"),
        
        ("radiology.yml", 
         "Radiologist with MD degree from U.S. or India, experience reading CT and MRI scans, diagnostic workflows, AI-assisted image analysis, board certification"),
        
        ("doctors_md.yml", 
         "U.S.-trained physician MD from top U.S. medical school with 2+ years clinical practice experience as General Practitioner, chronic care management, telemedicine"),
        
        ("biology_expert.yml", 
         "Biologist with PhD from top U.S. university, undergraduate U.S. U.K. or Canada, specializing in molecular biology, gene expression, genetics, CRISPR, PCR, sequencing"),
        
        ("anthropology.yml", 
         "PhD student or completed PhD in anthropology from distinguished U.S. university, focused on labor migration, cultural identity, ethnographic methods, fieldwork, sociology"),
        
        ("mathematics_phd.yml", 
         "Mathematician with PhD from leading U.S. university, undergraduate U.S. U.K. or Canada, specializing in statistical inference, stochastic processes, mathematics or statistics"),
        
        ("quantitative_finance.yml", 
         "MBA from prestigious U.S. university M7 MBA with 3+ years experience in quantitative finance, risk modeling, algorithmic trading, Python, portfolio optimization, derivatives pricing"),
        
        ("bankers.yml", 
         "Healthcare investment banker MBA from U.S. university with 2+ years in investment banking, M&A advisory, healthcare-focused, biotech, pharma services, private equity"),
        
        ("mechanical_engineers.yml", 
         "Mechanical engineer with higher degree and 3+ years experience in product development, structural design, SolidWorks, ANSYS, thermal systems, CAD tools, prototyping"),
    ]


def process_query(config_path, query_text, tpuf_client, vo_client, reranker):
    """
    Execute complete search pipeline for a single query.
    
    Args:
        config_path (str): Configuration file name
        query_text (str): Search query
        tpuf_client: Turbopuffer client
        vo_client: Voyage AI client
        reranker: Cross-encoder model
        
    Returns:
        dict: Processing result with status and evaluation response
    """
    try:
        candidates = search_candidates(query_text, tpuf_client, vo_client)
        
        if not candidates:
            logger.warning(f"No candidates found for {config_path}")
            return {'config': config_path, 'status': 'NO_RESULTS'}
        
        reranked = rerank_candidates(query_text, candidates, reranker)
        object_ids = [extract_row_id(c) for c in reranked]
        
        logger.info(f"Submitting {len(object_ids)} candidates for evaluation")
        result = submit_evaluation(config_path, object_ids)
        
        logger.info(f"Evaluation completed for {config_path}")
        return {'config': config_path, 'status': 'SUCCESS', 'result': result}
        
    except Exception as e:
        logger.error(f"Error processing {config_path}: {str(e)}", exc_info=True)
        return {'config': config_path, 'status': 'FAILED', 'error': str(e)}


def main():
    """Main execution function."""
    logger.info("Starting candidate search and re-ranking system")
    
    tpuf_client, vo_client, reranker = initialize_clients()
    queries = define_queries()
    
    logger.info(f"Processing {len(queries)} query configurations")
    print("=" * 70)
    
    results_summary = []
    
    for i, (config_path, query_text) in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Processing: {config_path}")
        print("-" * 70)
        
        result = process_query(config_path, query_text, tpuf_client, vo_client, reranker)
        results_summary.append(result)
        
        if result['status'] == 'SUCCESS':
            print(f"Status: SUCCESS")
            if 'average_final_score' in result.get('result', {}):
                score = result['result']['average_final_score']
                print(f"Average Score: {score:.2f}/100")
        else:
            print(f"Status: {result['status']}")
    
    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    
    success_count = sum(1 for r in results_summary if r['status'] == 'SUCCESS')
    print(f"Successful: {success_count}/{len(queries)}")
    print(f"Failed: {len(queries) - success_count}/{len(queries)}\n")
    
    for result in results_summary:
        status = "SUCCESS" if result['status'] == 'SUCCESS' else result['status']
        print(f"[{status}] {result['config']}")
    
    print("=" * 70)
    logger.info("All queries processed successfully")


if __name__ == "__main__":
    main()
