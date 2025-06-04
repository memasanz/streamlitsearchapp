# Streamlit Search App

A Streamlit-based intelligent chat application that integrates Azure OpenAI and Azure Cognitive Search services to provide document-based question answering capabilities.

## ⚠️ Disclaimer

**This is sample/demonstration code only and is not intended for production use.** This application is provided for educational and testing purposes to showcase integration between Azure OpenAI and Azure Cognitive Search services. For production deployments, please ensure proper security measures, error handling, monitoring, and compliance with your organization's requirements.

## Features

- **Direct Chat Interface**: Direct conversation with Azure OpenAI GPT models
- **Document Search & Chat**: Question-answering based on indexed documents using Azure Cognitive Search
- **Multi-Purpose Templates**: Pre-configured for antenna parts selection and IT helpdesk scenarios
- **Configuration Management**: Web-based settings panel for Azure service configuration
- **Index Management**: Built-in tools to create and manage Azure Cognitive Search indexes

## Application Structure

### Main Components

1. **Home**: Basic chat interface using Azure OpenAI directly
2. **Settings**: Configuration panel for Azure services and index management
3. **Upload File**: File upload functionality (placeholder)
4. **Chat**: Advanced chat with document search capabilities using Cognitive Search

### Core Files

- `webapp.py`: Main Streamlit application with user interface
- `cog_search.py`: Azure Cognitive Search integration and OpenAI helpers
- `credentials.py`: Configuration settings and service credentials
- `requirements.txt`: Python dependencies

## Prerequisites

### Azure Services Required

1. **Azure OpenAI Service**
   - Deployed GPT-35-Turbo model
   - Text-Davinci-003 model
   - API endpoint and access keys

2. **Azure Cognitive Search**
   - Search service instance
   - API keys with admin permissions

3. **Azure Storage Account**
   - Blob storage container for documents
   - Connection string and access keys

4. **Azure Cognitive Services**
   - Multi-service cognitive services resource
   - API keys for text processing

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/memasanz/streamlitsearchapp.git
   cd streamlitsearchapp
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit
   pip install streamlit-chat
   pip install streamlit-option-menu
   pip install openai
   pip install langchain
   pip install faiss-cpu
   pip install transformers
   pip install pandas
   pip install numpy
   pip install requests
   ```

3. **Configure Azure services**
   
   Edit `credentials.py` with your Azure service details:

   ```python
   # Azure OpenAI Configuration
   AZURE_OPENAI_ENDPOINT = "https://your-openai-service.openai.azure.com/"
   AZURE_OPENAI_KEY = "your-openai-api-key"
   
   # Azure Cognitive Search Configuration
   COG_SEARCH_RESOURCE = "your-search-service-name"
   COG_SEARCH_KEY = "your-search-api-key"
   COG_SEARCH_INDEX = "your-index-name"
   
   # Azure Storage Configuration
   STORAGE_CONNECTION_STRING = "your-storage-connection-string"
   STORAGE_ACCOUNT = "your-storage-account-name"
   STORAGE_KEY = "your-storage-key"
   ```

## Usage

### Running the Application

```bash
streamlit run webapp.py
```

The application will start on `http://localhost:8501`

### Initial Setup

1. **Configure Services**: Navigate to the "Settings" tab and enter your Azure service credentials
2. **Create Search Index**: Use the "Create Index" button to set up your Cognitive Search index
3. **Upload Documents**: Upload documents to your Azure Storage container for indexing

### Using the Chat Features

#### Direct Chat (Home Tab)
- Simple conversation interface with Azure OpenAI
- Uses GPT-35-Turbo model for responses
- No document context - general purpose chat

#### Document-Based Chat (Chat Tab)
- Ask questions about your uploaded documents
- Uses Cognitive Search to find relevant content
- Combines search results with OpenAI for comprehensive answers

## Configuration Options

### Question Templates

The application includes pre-configured templates for specific use cases:

#### Antenna Parts Selector
```python
XTG_TEMPPLATE_ORIG = """
You are an antenna part selector chat bot.  
You are given sections of a catalog of antennas.
Each document section includes the part number at the beginning...
"""
```

#### IT Helpdesk
```python
TTEC_questiontemplate = """
You are an IT HelpDesk Chat Bot.  
You are to answer questions based on the context provided...
"""
```

### Model Configuration

- **GPT-35-Turbo**: For chat conversations
- **Text-Davinci-003**: For document-based Q&A
- **Text-Search-Curie**: For embeddings and similarity search

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   Azure OpenAI   │────│  Azure Cognitive│
│                 │    │                  │    │     Search      │
│ - Chat Interface│    │ - GPT Models     │    │                 │
│ - Settings      │    │ - Embeddings     │    │ - Document Index│
│ - File Upload   │    │ - Text Generation│    │ - Search & Rank │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                      ┌──────────────────┐
                      │  Azure Storage   │
                      │                  │
                      │ - Document Store │
                      │ - Blob Container │
                      └──────────────────┘
```

## Advanced Features

### Index Management

The application provides built-in tools to:
- Create data sources pointing to your Azure Storage
- Set up cognitive skillsets for document processing
- Create and configure search indexes
- Run indexers to process and index documents

### Session Management

- Chat history is maintained in Streamlit session state
- Settings can be updated dynamically
- Message reset functionality available

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify Azure service credentials in `credentials.py`
   - Ensure API keys have proper permissions
   - Check endpoint URLs are correct

2. **Search Not Working**
   - Confirm Cognitive Search index exists
   - Verify documents are uploaded to storage container
   - Check indexer has run successfully

3. **Chat Responses Incomplete**
   - Increase `max_tokens` in OpenAI configuration
   - Verify model deployment names match configuration

### Debug Mode

Set `DEBUG = "1"` in `credentials.py` to enable additional logging.

## Security Considerations

- Never commit actual API keys to version control
- Use environment variables for production deployments
- Regularly rotate API keys and connection strings
- Implement proper access controls for Azure resources

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is provided as-is for demonstration and educational purposes only. **This code is not production-ready and should not be used in production environments** without significant modifications, testing, and security enhancements. Please ensure compliance with Azure service terms and conditions.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Azure service documentation
3. Open an issue in the GitHub repository