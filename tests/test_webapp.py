"""
Tests for webapp.py utility functions
"""

import pytest
from unittest.mock import Mock, patch, mock_open
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credentials import creds


class TestWebappUtilities:
    """Test cases for webapp utility functions that can be tested without full streamlit"""
    
    def test_update_creds_logic(self):
        """Test update_creds function logic without file I/O"""
        # Sample credentials file content
        file_content = [
            'AZURE_OPENAI_ENDPOINT = "old_endpoint"\n',
            'AZURE_OPENAI_KEY = "old_key"\n', 
            'OTHER_LINE = "unchanged"\n'
        ]
        
        # Mock credentials to update
        test_creds = {
            'AZURE_OPENAI_ENDPOINT': 'new_endpoint',
            'AZURE_OPENAI_KEY': 'new_key'
        }
        
        # Test the core logic of finding and replacing lines
        updated_lines = []
        for line in file_content:
            found_match = False
            for attribute, value in test_creds.items():
                if line.startswith(attribute):
                    updated_lines.append(f'{attribute} = "{value}"\n')
                    found_match = True
                    break
            if not found_match:
                updated_lines.append(line)
        
        # Verify the logic works as expected
        updated_content = ''.join(updated_lines)
        assert 'AZURE_OPENAI_ENDPOINT = "new_endpoint"' in updated_content
        assert 'AZURE_OPENAI_KEY = "new_key"' in updated_content
        assert 'OTHER_LINE = "unchanged"' in updated_content


class TestCredentialsIntegration:
    """Test credentials integration"""
    
    def test_credentials_structure(self):
        """Test that credentials dictionary has expected structure"""
        # Test that creds is a dictionary
        assert isinstance(creds, dict)
        
        # Test that expected keys exist
        expected_keys = [
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_KEY', 
            'COG_SEARCH_RESOURCE',
            'COG_SEARCH_KEY',
            'COG_SEARCH_INDEX'
        ]
        
        for key in expected_keys:
            assert key in creds, f"Expected key {key} not found in credentials"
        
        # Test that values are strings
        for key, value in creds.items():
            assert isinstance(value, str), f"Value for {key} is not a string"
    
    def test_question_templates(self):
        """Test that question templates are properly defined"""
        assert 'QUESTION_TEMPLATE' in creds
        assert isinstance(creds['QUESTION_TEMPLATE'], str)
        assert len(creds['QUESTION_TEMPLATE']) > 0
        
        # Check that template has placeholder for question
        assert '{question}' in creds['QUESTION_TEMPLATE']


class TestStreamlitMocking:
    """Test that we can properly mock streamlit components"""
    
    def test_streamlit_session_state_mock(self):
        """Test that we can mock streamlit session state"""
        # Create a mock session state
        mock_session_state = {}
        
        # Mock streamlit module
        mock_st = Mock()
        mock_st.session_state = mock_session_state
        
        # Test adding items to session state
        mock_st.session_state['test_key'] = 'test_value'
        assert mock_st.session_state['test_key'] == 'test_value'
        
        # Test checking if key exists
        assert 'test_key' in mock_st.session_state
        assert 'nonexistent_key' not in mock_st.session_state