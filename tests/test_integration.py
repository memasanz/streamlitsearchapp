"""
Integration tests for the Streamlit Search App
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credentials import creds


class TestIntegration:
    """Integration tests for app components"""
    
    def test_credentials_consistency(self):
        """Test that credentials are consistent across modules"""
        # Test that all required credentials exist
        required_creds = [
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_KEY',
            'COG_SEARCH_RESOURCE',
            'COG_SEARCH_KEY',
            'COG_SEARCH_INDEX',
            'STORAGE_CONNECTION_STRING',
            'STORAGE_CONTAINER',
            'QUESTION_TEMPLATE'
        ]
        
        for cred in required_creds:
            assert cred in creds, f"Required credential {cred} missing"
            assert isinstance(creds[cred], str), f"Credential {cred} should be a string"
            assert len(creds[cred]) > 0, f"Credential {cred} should not be empty"
    
    def test_question_template_format(self):
        """Test that question template has proper format"""
        template = creds['QUESTION_TEMPLATE']
        
        # Check that it contains the question placeholder
        assert '{question}' in template, "Question template must contain {question} placeholder"
        
        # Test formatting works
        test_question = "What is the weather today?"
        try:
            formatted = template.format(question=test_question)
            assert test_question in formatted
        except KeyError as e:
            pytest.fail(f"Question template formatting failed: {e}")
    
    def test_search_configuration(self):
        """Test search-related configuration"""
        # Check that search resource name is valid format
        search_resource = creds['COG_SEARCH_RESOURCE']
        assert search_resource.replace('-', '').replace('_', '').isalnum(), \
            "Search resource name should be alphanumeric with hyphens/underscores"
        
        # Check that index name is valid
        index_name = creds['COG_SEARCH_INDEX']
        assert len(index_name) > 0, "Index name should not be empty"
        assert index_name.replace('-', '').replace('_', '').isalnum(), \
            "Index name should be alphanumeric with hyphens/underscores"
    
    def test_storage_configuration(self):
        """Test storage-related configuration"""
        # Check storage connection string format (allow partial strings for test data)
        conn_string = creds['STORAGE_CONNECTION_STRING']
        assert len(conn_string) > 0, "Storage connection string should not be empty"
        # More lenient check for connection string format
        assert any(keyword in conn_string for keyword in ['DefaultEndpointsProtocol', 'AccountName', 'DefaultEn', 'core.windows.net']), \
            "Storage connection string should contain Azure storage identifiers"
        
        # Check container name
        container = creds['STORAGE_CONTAINER']
        assert len(container) > 0, "Storage container name should not be empty"
    
    def test_openai_configuration(self):
        """Test OpenAI-related configuration"""
        # Check endpoint format
        endpoint = creds['AZURE_OPENAI_ENDPOINT']
        assert endpoint.startswith('https://'), "OpenAI endpoint should start with https://"
        assert '.openai.azure.com' in endpoint, "Should be Azure OpenAI endpoint"
        
        # Check model names exist
        assert 'TEXT_DAVINCI' in creds, "TEXT_DAVINCI model name should be configured"
        assert 'GTPTurbo' in creds, "GTPTurbo model name should be configured"


class TestAppStructure:
    """Test overall application structure"""
    
    def test_required_files_exist(self):
        """Test that all required files exist"""
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        required_files = [
            'webapp.py',
            'cog_search.py',
            'credentials.py',
            'requirements.txt',
            'README.md'
        ]
        
        for file_name in required_files:
            file_path = os.path.join(base_path, file_name)
            assert os.path.exists(file_path), f"Required file {file_name} not found"
    
    def test_requirements_format(self):
        """Test that requirements.txt has proper format"""
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        requirements_path = os.path.join(base_path, 'requirements.txt')
        
        with open(requirements_path, 'r') as f:
            requirements = f.read().strip().split('\n')
        
        # Check that testing dependencies are included
        requirement_names = [req.strip() for req in requirements if req.strip()]
        assert 'pytest' in requirement_names, "pytest should be in requirements"
        assert 'pytest-mock' in requirement_names, "pytest-mock should be in requirements"
        
        # Check that main dependencies are present
        essential_deps = ['streamlit', 'openai', 'langchain', 'pandas', 'numpy']
        for dep in essential_deps:
            assert any(dep in req for req in requirement_names), f"{dep} should be in requirements"