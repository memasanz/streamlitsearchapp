"""
Tests for cog_search.py classes and methods
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credentials import creds


class TestCogSearchHelper:
    """Test cases for CogSearchHelper class"""
    
    def setup_method(self):
        """Setup method called before each test"""
        # Mock the dependencies before importing
        self.mock_modules = {
            'langchain_community.llms': Mock(),
            'langchain_community.vectorstores': Mock(),
            'langchain_community.embeddings': Mock(),
            'langchain.chains.question_answering': Mock(),
            'openai': Mock(),
            'transformers': Mock(),
            'pandas': Mock()
        }
        
        for module_name, mock_module in self.mock_modules.items():
            sys.modules[module_name] = mock_module
    
    def test_init_with_default_index(self):
        """Test CogSearchHelper initialization with default index"""
        with patch.dict('sys.modules', self.mock_modules):
            from cog_search import CogSearchHelper
            
            helper = CogSearchHelper(index=None)
            
            assert helper.service_name == creds['COG_SEARCH_RESOURCE']
            assert helper.search_key == creds['COG_SEARCH_KEY']
            assert helper.storage_connectionstring == creds['STORAGE_CONNECTION_STRING']
            assert helper.storage_container == creds['STORAGE_CONTAINER']
            assert helper.cognitive_service_key == creds['COG_SERVICE_KEY']
            assert helper.index == creds['COG_SEARCH_INDEX']
    
    def test_init_with_custom_index(self):
        """Test CogSearchHelper initialization with custom index"""
        with patch.dict('sys.modules', self.mock_modules):
            from cog_search import CogSearchHelper
            
            custom_index = "custom_test_index"
            helper = CogSearchHelper(index=custom_index)
            
            assert helper.index == custom_index
            assert helper.service_name == creds['COG_SEARCH_RESOURCE']
    
    def test_get_the_token_count_success(self):
        """Test token count calculation with valid input"""
        with patch.dict('sys.modules', self.mock_modules):
            from cog_search import CogSearchHelper
            
            # Mock the tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = ['token1', 'token2', 'token3']
            
            with patch('cog_search.GPT2TokenizerFast') as mock_tokenizer_class:
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
                
                helper = CogSearchHelper(index="test")
                result = helper.get_the_token_count("test document")
                
                assert result == 3
                mock_tokenizer.encode.assert_called_once_with("test document")
    
    def test_get_the_token_count_failure(self):
        """Test token count calculation with encoding failure"""
        with patch.dict('sys.modules', self.mock_modules):
            from cog_search import CogSearchHelper
            
            # Mock the tokenizer to raise an exception
            mock_tokenizer = Mock()
            mock_tokenizer.encode.side_effect = Exception("Encoding failed")
            
            with patch('cog_search.GPT2TokenizerFast') as mock_tokenizer_class:
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
                
                helper = CogSearchHelper(index="test")
                
                # Capture print output
                with patch('builtins.print') as mock_print:
                    result = helper.get_the_token_count("test document")
                
                assert result == -1
                mock_print.assert_called_once_with('failed to get token count')
    
    def test_create_datasource_payload(self):
        """Test create_datasource request payload structure"""
        with patch.dict('sys.modules', self.mock_modules):
            from cog_search import CogSearchHelper
            
            with patch('cog_search.requests') as mock_requests:
                mock_response = Mock()
                mock_response.status_code = 201
                mock_requests.request.return_value = mock_response
                
                helper = CogSearchHelper(index="test_index")
                response, success = helper.create_datasource()
                
                # Verify that requests.request was called
                mock_requests.request.assert_called_once()
                call_args = mock_requests.request.call_args
                
                # Check method and URL
                assert call_args[0][0] == "PUT"
                expected_url = f"https://{creds['COG_SEARCH_RESOURCE']}.search.windows.net//datasources/test_index-datasource?api-version=2020-06-30"
                assert call_args[0][1] == expected_url
                
                # Check headers
                headers = call_args[1]['headers']
                assert headers['api-key'] == creds['COG_SEARCH_KEY']
                assert headers['Content-Type'] == 'application/json'
                
                # Check payload structure
                payload_str = call_args[1]['data']
                payload = json.loads(payload_str)
                
                assert payload['type'] == 'azureblob'
                assert payload['credentials']['connectionString'] == creds['STORAGE_CONNECTION_STRING']
                assert payload['container']['name'] == creds['STORAGE_CONTAINER']
                
                # Check return values
                assert success is True
                assert response == mock_response
    
    def test_create_datasource_failure(self):
        """Test create_datasource with failure response"""
        with patch.dict('sys.modules', self.mock_modules):
            from cog_search import CogSearchHelper
            
            with patch('cog_search.requests') as mock_requests:
                mock_response = Mock()
                mock_response.status_code = 400  # Bad request
                mock_requests.request.return_value = mock_response
                
                helper = CogSearchHelper(index="test_index")
                response, success = helper.create_datasource()
                
                assert success is False
                assert response == mock_response
    
    def test_create_datasource_success_204(self):
        """Test create_datasource with 204 status code (also success)"""
        with patch.dict('sys.modules', self.mock_modules):
            from cog_search import CogSearchHelper
            
            with patch('cog_search.requests') as mock_requests:
                mock_response = Mock()
                mock_response.status_code = 204  # No content (success)
                mock_requests.request.return_value = mock_response
                
                helper = CogSearchHelper(index="test_index")
                response, success = helper.create_datasource()
                
                assert success is True
                assert response == mock_response


class TestOpenAIHelper:
    """Test cases for OpenAIHelper class"""
    
    def setup_method(self):
        """Setup method called before each test"""
        # Mock the dependencies before importing
        self.mock_modules = {
            'langchain_community.llms': Mock(),
            'langchain_community.vectorstores': Mock(),
            'langchain_community.embeddings': Mock(),
            'langchain.chains.question_answering': Mock(),
            'openai': Mock(),
            'transformers': Mock(),
            'pandas': Mock()
        }
        
        for module_name, mock_module in self.mock_modules.items():
            sys.modules[module_name] = mock_module
    
    def test_openai_helper_init(self):
        """Test OpenAIHelper initialization"""
        with patch.dict('sys.modules', self.mock_modules):
            from cog_search import OpenAIHelper
            
            helper = OpenAIHelper(index="test_index")
            assert helper.index == "test_index"
    
    def test_get_the_token_count_inherited(self):
        """Test that OpenAIHelper inherits token count method from CogSearchHelper"""
        with patch.dict('sys.modules', self.mock_modules):
            from cog_search import OpenAIHelper
            
            # Mock the tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = ['token1', 'token2']
            
            with patch('cog_search.GPT2TokenizerFast') as mock_tokenizer_class:
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
                
                helper = OpenAIHelper(index="test")
                result = helper.get_the_token_count("test")
                
                assert result == 2


class TestUtilityFunctions:
    """Test utility functions and helpers"""
    
    def test_credentials_loaded(self):
        """Test that credentials are properly loaded"""
        assert isinstance(creds, dict)
        assert len(creds) > 0
        
        required_keys = [
            'COG_SEARCH_RESOURCE',
            'COG_SEARCH_KEY',
            'STORAGE_CONNECTION_STRING',
            'STORAGE_CONTAINER'
        ]
        
        for key in required_keys:
            assert key in creds, f"Required credential {key} not found"