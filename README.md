# Smart Sales Email Generator

An AI-powered tool that generates contextual and professional follow-up emails based on previous customer interactions, using RAG and sentiment analysis.

## AI Tools & Technologies
- RAG (Retrieval Augmented Generation) Implementation:
  * Vector Store: ChromaDB for email template storage
  * Embeddings: HuggingFace Sentence Transformers
  * Similarity Search for context retrieval
- LangChain for orchestrating the RAG pipeline
- Hugging Face Transformers for sentiment analysis
- DeepSeek model for email generation
- Gradio for the interactive web interface
- Transformers pipeline for NLP tasks

## Key Features
- RAG-powered contextual email generation
- Retrieval of similar past interactions
- Automated sentiment analysis for tone detection
- Customizable urgency levels and situation types
- Real-time email quality scoring
- Multiple pre-built templates for common scenarios
- Context-aware response generation

## Technical Skills Demonstrated
- RAG System Implementation
- Vector Database Management
- Embedding Generation
- Natural Language Processing (NLP)
- Large Language Model (LLM) integration
- Prompt engineering
- API integration (Hugging Face Hub)
- Web application development
- Machine Learning model deployment
- GPU acceleration support
- Error handling and input validation

## Architecture
- RAG Components:
  * Vector Store for template storage
  * Embedding model for text vectorization
  * Similarity search for context retrieval
- Language Models:
  * DeepSeek for generation
  * BERT-based model for sentiment analysis
- Interface:
  * Gradio for web UI
  * Real-time processing

## Use Cases
- Customer Service Follow-ups
- Complaint Resolution
- Service Issue Communication
- Payment Dispute Handling
- Product Query Responses
- General Business Communication

## How to Use
1. Enter the previous customer interaction
2. Select the situation type from available options
3. Choose tone (optional - will be automatically detected)
4. Set urgency level (High/Medium/Low)
5. Submit to generate a professional follow-up email with quality score


## Development Stack
- Python 3.x
- LangChain Framework
- ChromaDB
- HuggingFace Transformers
- Gradio UI Framework
- CUDA support for GPU acceleration

## Future Enhancements
- Enhanced RAG capabilities
- Expanded template database
- Response time optimization
- Direct email system integration
- Analytics and tracking capabilities
- Enhanced scoring system

## License
MIT License

## Author
[Tobi Ajibola]
