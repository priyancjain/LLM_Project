# PDF Question Answering System

This project implements a PDF Question Answering system using LangChain, Ollama, and Streamlit. It allows users to upload PDF documents and ask questions about their content, getting AI-powered responses based on the document's content.

## Why Ollama?

This project uses Ollama instead of cloud-based APIs for several key advantages:

1. **Privacy & Security**
   - All processing happens locally on your machine
   - No data is sent to external servers
   - Complete control over your documents and queries

2. **Cost-Effective**
   - No API usage fees or subscription costs
   - No token-based pricing
   - Free to use without limitations

3. **Offline Capability**
   - Works without internet connection
   - No dependency on external services
   - Consistent performance regardless of network status

4. **Customization**
   - Ability to use different models locally
   - Custom model fine-tuning possibilities
   - Full control over model parameters

5. **Performance**
   - Lower latency as everything runs locally
   - No rate limiting or API quotas
   - Consistent response times

## Features

- PDF document upload and processing
- Document chunking and vectorization using GPT4All embeddings
- Question answering using Llama2 model through Ollama
- Interactive web interface built with Streamlit
- History of previous queries and answers
- Concise and accurate responses

## Prerequisites

- Python 3.8 or higher
- Ollama installed locally
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:

For macOS/Linux:
```bash
python -m venv env
source env/bin/activate
```

For Windows:
```bash
python -m venv env
.\env\Scripts\activate
```

3. Install Ollama:
   - Visit [Ollama's official website](https://ollama.ai/) and download the appropriate version for your OS
   - Follow the installation instructions for your platform
   - Pull the Llama2 model:
   ```bash
   ollama pull llama2:7b
   ```

4. Install project dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload a PDF file using the file uploader

4. Click "Process PDF" to initialize the document processing

5. Enter your questions in the query input field and click "Submit Query"

6. View the answers and previous conversation history below

## Project Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Project dependencies
- `uploaded_file.pdf`: Temporary storage for uploaded PDFs
- `my_vectorstore.pkl`: Vector store for document embeddings

## Future Goals

1. **Enhanced Document Processing**
   - Support for multiple document formats (DOCX, TXT, etc.)
   - Batch processing of multiple documents
   - Improved text chunking strategies

2. **Advanced Features**
   - Document summarization
   - Key points extraction
   - Citation tracking for answers
   - Support for images and tables in documents

3. **UI/UX Improvements**
   - Dark mode support
   - Better error handling and user feedback
   - Export conversation history
   - Customizable response length and style

4. **Performance Optimization**
   - Caching mechanisms for faster responses
   - Optimized vector storage
   - Support for larger documents

5. **Model Enhancements**
   - Support for multiple LLM models
   - Fine-tuning capabilities
   - Custom model integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
