<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>README - Mercor Search Engineering Assignment</title>
</head>
<body>
<h1>Mercor Search Engineering Assignment</h1>

<h2>Overview</h2>
<p>Two-stage candidate search and re-ranking system using vector search and cross-encoder re-ranking.</p>

<h2>Architecture</h2>

<h3>Stage 1: Vector Search</h3>
<ul>
    <li>Turbopuffer vector database with Voyage-3 embeddings (1024 dims)</li>
    <li>Retrieves top 50 candidates via ANN search</li>
</ul>

<h3>Stage 2: Cross-Encoder Re-ranking</h3>
<ul>
    <li>ms-marco-MiniLM-L-6-v2 model</li>
    <li>Re-ranks candidates for relevance</li>
    <li>Returns top 10 for evaluation</li>
</ul>

<h2>Technical Stack</h2>
<ul>
    <li><strong>Vector Database:</strong> Turbopuffer (aws-us-west-2)</li>
    <li><strong>Embeddings:</strong> Voyage AI (voyage-3 model)</li>
    <li><strong>Re-ranker:</strong> Sentence Transformers cross-encoder</li>
    <li><strong>Language:</strong> Python 3.x</li>
</ul>

<h2>Setup & Execution</h2>
<pre><code># Install dependencies
pip install turbopuffer voyageai sentence-transformers requests
#Run the system
python search.py</code></pre>
<h2>Results Summary</h2>
<p>Successfully processed all 10 query configurations:</p>
<ul>
    <li>Tax Lawyer: 77.3/100</li>
    <li>Junior Corporate Lawyer: 69.7/100</li>
    <li>Radiology: 44.3/100</li>
    <li>[See results.txt for complete details]</li>
</ul>

<h2>Key Features</h2>
<ul>
    <li>Robust error handling with logging</li>
    <li>Flexible attribute extraction from Turbopuffer rows</li>
    <li>Efficient batch processing of queries</li>
    <li>Professional code structure with documentation</li>
</ul>

<h2>Files Included</h2>
<ul>
    <li><code>search.py</code> - Main implementation</li>
    <li><code>results.txt</code> - Complete execution output with scores</li>
    <li><code>README.md</code> - This file</li>
    <li><code>requirements.txt</code> - Python dependencies</li>
</ul>

<h2>Future Improvements</h2>
<ul>
    <li>Hard criteria pre-filtering using metadata</li>
    <li>Query-specific filter rules</li>
    <li>Hybrid search (BM25 + vector)</li>
    <li>Fine-tuned cross-encoder for domain-specific ranking</li>
</ul>
