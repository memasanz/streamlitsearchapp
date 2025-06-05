# Tests for Streamlit Search App

This directory contains comprehensive unit and integration tests for the Streamlit Search App.

## Test Structure

### `test_webapp.py`
Tests for utility functions in `webapp.py`:
- Credential management logic
- Configuration validation
- Streamlit integration helpers

### `test_cog_search.py`  
Tests for core classes in `cog_search.py`:
- `CogSearchHelper` class initialization and methods
- `OpenAIHelper` class functionality
- Token counting utilities
- Azure Cognitive Search API request formatting

### `test_integration.py`
Integration tests ensuring components work together:
- Configuration consistency across modules
- File structure validation
- Service configuration validation
- Requirements format verification

## Running Tests

### Run all tests:
```bash
python -m pytest tests/ -v
```

### Run specific test file:
```bash
python -m pytest tests/test_webapp.py -v
```

### Run with coverage (if coverage is installed):
```bash
python -m pytest tests/ --cov=. --cov-report=html
```

## Test Configuration

- `pytest.ini` - Test configuration and settings
- Tests use mocking to avoid external API calls
- Warnings are filtered for cleaner output

## Test Coverage

The tests cover:
- ✅ Core class initialization
- ✅ Configuration management  
- ✅ Token counting functionality
- ✅ Azure API request formatting
- ✅ Error handling
- ✅ Integration between components
- ✅ File structure validation

## Adding New Tests

When adding new functionality to the app:

1. Add unit tests for new functions/methods
2. Add integration tests for new workflows
3. Mock external dependencies (Azure APIs, OpenAI, etc.)
4. Follow existing test patterns and naming conventions

## Notes

- Tests are designed to run without external dependencies
- Azure API calls are mocked to prevent actual service calls
- Configuration uses the existing `credentials.py` values
- Some warnings about numpy reloading are expected and filtered