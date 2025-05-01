# Course-APP
Course App Chatbot for National Education Policy Project
<br>
<ul>
  <li>Vector DB: Pinecone</li>
  <li>Vector Dimensions: 768</li>
  <li>Similarity Metrics: cosine similarity</li>
  <li>LLM: llama-3.1-8b-instant via Groq API</li>
  <li>temperature=0.5</li>
  <li>max_tokens=1024</li>
</ul>

download embedding model (768 dims) by running:  
`encoder = HuggingFaceEncoder(name = "dwzhu/e5-base-4k")`
