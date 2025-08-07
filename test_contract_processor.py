"""
Test Suite for Legal Contract Processing Pipeline
===============================================

Comprehensive tests for all components of the contract processing pipeline.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
import json

# Import modules to test
from contract_processor import (
    ContractData, 
    DocumentProcessor, 
    LLMProcessor, 
    SemanticSearch,
    ContractProcessingPipeline
)

class TestContractData(unittest.TestCase):
    """Test ContractData dataclass"""
    
    def test_contract_data_creation(self):
        """Test basic contract data creation"""
        contract = ContractData(
            contract_id="test_001",
            filename="test.pdf",
            text="Sample contract text"
        )
        
        self.assertEqual(contract.contract_id, "test_001")
        self.assertEqual(contract.filename, "test.pdf")
        self.assertEqual(contract.text, "Sample contract text")
        self.assertEqual(contract.summary, "")
        self.assertIsNone(contract.extraction_metadata)


class TestDocumentProcessor(unittest.TestCase):
    """Test DocumentProcessor class"""
    
    def setUp(self):
        self.processor = DocumentProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_normalize_text(self):
        """Test text normalization"""
        raw_text = "  This   is    sample   text  with  extra   spaces  \n\n  Page 123  \n"
        normalized = self.processor.normalize_text(raw_text)
        
        self.assertNotIn("  ", normalized)  # No double spaces
        self.assertNotIn("Page 123", normalized)  # Page numbers removed
        self.assertTrue(len(normalized) > 0)
    
    def test_normalize_text_special_chars(self):
        """Test normalization handles special characters"""
        raw_text = "Contract™ with ® symbols and © marks"
        normalized = self.processor.normalize_text(raw_text)
        
        # Should remove special trademark symbols
        self.assertNotIn("™", normalized)
        self.assertNotIn("®", normalized)
        self.assertNotIn("©", normalized)
    
    @patch('contract_processor.fitz')
    def test_extract_text_from_pdf_success(self, mock_fitz):
        """Test successful PDF text extraction"""
        # Mock fitz document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Sample PDF text"
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_fitz.open.return_value = mock_doc
        
        result = self.processor.extract_text_from_pdf("test.pdf")
        
        self.assertEqual(result, "Sample PDF text")
        mock_fitz.open.assert_called_once_with("test.pdf")
        mock_doc.close.assert_called_once()
    
    def test_load_contracts_empty_directory(self):
        """Test loading contracts from empty directory"""
        contracts = self.processor.load_contracts(self.temp_dir, max_contracts=10)
        self.assertEqual(len(contracts), 0)
    
    def test_load_contracts_nonexistent_directory(self):
        """Test loading from non-existent directory raises error"""
        with self.assertRaises(FileNotFoundError):
            self.processor.load_contracts("/nonexistent/path")


class TestLLMProcessor(unittest.TestCase):
    """Test LLMProcessor class"""
    
    def setUp(self):
        self.processor = LLMProcessor(api_key="test-key")
    
    def test_few_shot_examples_loaded(self):
        """Test that few-shot examples are properly loaded"""
        self.assertIn("termination", self.processor.few_shot_examples)
        self.assertIn("confidentiality", self.processor.few_shot_examples)
        self.assertIn("liability", self.processor.few_shot_examples)
        
        for example in self.processor.few_shot_examples.values():
            self.assertTrue(len(example) > 0)
    
    def test_create_clause_extraction_prompt(self):
        """Test clause extraction prompt creation"""
        text = "Sample contract text with termination clause"
        prompt = self.processor._create_clause_extraction_prompt(text, "termination")
        
        self.assertIn("termination", prompt.lower())
        self.assertIn("Sample contract text", prompt)
        self.assertIn("Extract", prompt)
    
    def test_create_summary_prompt(self):
        """Test summary prompt creation"""
        text = "Sample contract text for summarization"
        prompt = self.processor._create_summary_prompt(text)
        
        self.assertIn("100-150 words", prompt)
        self.assertIn("Purpose", prompt)
        self.assertIn("obligations", prompt)
        self.assertIn("Sample contract text", prompt)
    
    @patch('contract_processor.openai.ChatCompletion.create')
    def test_extract_clause_success(self, mock_openai):
        """Test successful clause extraction"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Extracted termination clause text"
        mock_openai.return_value = mock_response
        
        result = self.processor.extract_clause("Sample contract", "termination")
        
        self.assertEqual(result, "Extracted termination clause text")
        mock_openai.assert_called_once()
    
    @patch('contract_processor.openai.ChatCompletion.create')
    def test_extract_clause_api_error(self, mock_openai):
        """Test clause extraction handles API errors"""
        mock_openai.side_effect = Exception("API Error")
        
        result = self.processor.extract_clause("Sample contract", "termination")
        
        self.assertIn("Error extracting termination clause", result)
    
    @patch('contract_processor.openai.ChatCompletion.create')
    def test_generate_summary_success(self, mock_openai):
        """Test successful summary generation"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated contract summary"
        mock_openai.return_value = mock_response
        
        result = self.processor.generate_summary("Sample contract text")
        
        self.assertEqual(result, "Generated contract summary")
    
    def test_process_contract_integration(self):
        """Test complete contract processing (mocked)"""
        contract = ContractData("test_001", "test.pdf", "Sample contract text")
        
        with patch.object(self.processor, 'extract_clause') as mock_extract, \
             patch.object(self.processor, 'generate_summary') as mock_summary:
            
            mock_extract.return_value = "Mock clause"
            mock_summary.return_value = "Mock summary"
            
            result = self.processor.process_contract(contract)
            
            self.assertEqual(result.summary, "Mock summary")
            self.assertEqual(result.termination_clause, "Mock clause")
            self.assertEqual(result.confidentiality_clause, "Mock clause")
            self.assertEqual(result.liability_clause, "Mock clause")
            self.assertIsNotNone(result.extraction_metadata)


class TestSemanticSearch(unittest.TestCase):
    """Test SemanticSearch class"""
    
    def setUp(self):
        # Use a lightweight model for testing
        with patch('contract_processor.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_transformer.return_value = mock_model
            
            self.search = SemanticSearch()
            self.mock_model = mock_model
    
    def test_build_index_empty_contracts(self):
        """Test building index with empty contracts list"""
        self.search.build_index([])
        self.assertIsNone(self.search.index)
        self.assertEqual(len(self.search.clauses), 0)
    
    def test_build_index_with_contracts(self):
        """Test building index with valid contracts"""
        contracts = [
            ContractData("test_001", "test1.pdf", "text1", 
                        termination_clause="Termination clause 1",
                        confidentiality_clause="Confidentiality clause 1"),
            ContractData("test_002", "test2.pdf", "text2",
                        liability_clause="Liability clause 1")
        ]
        
        with patch('contract_processor.faiss') as mock_faiss:
            mock_index = MagicMock()
            mock_faiss.IndexFlatIP.return_value = mock_index
            mock_faiss.normalize_L2 = MagicMock()
            
            self.search.build_index(contracts)
            
            # Should have created index with clauses
            mock_faiss.IndexFlatIP.assert_called_once()
            self.assertEqual(len(self.search.clauses), 3)  # 3 non-empty clauses
    
    def test_search_no_index(self):
        """Test search without built index"""
        result = self.search.search("test query")
        self.assertEqual(result, [])


class TestContractProcessingPipeline(unittest.TestCase):
    """Test ContractProcessingPipeline class"""
    
    def setUp(self):
        with patch('contract_processor.openai') as mock_openai:
            self.pipeline = ContractProcessingPipeline(api_key="test-key")
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.assertIsNotNone(self.pipeline.doc_processor)
        self.assertIsNotNone(self.pipeline.llm_processor)
        self.assertIsNotNone(self.pipeline.semantic_search)
        self.assertIsNotNone(self.pipeline.logger)
    
    def test_save_results_csv_format(self):
        """Test saving results in CSV format"""
        contracts = [
            ContractData(
                contract_id="test_001",
                filename="test1.pdf",
                text="Sample text 1",
                summary="Summary 1",
                termination_clause="Termination 1",
                confidentiality_clause="Confidentiality 1",
                liability_clause="Liability 1",
                extraction_metadata={"text_length": 100, "word_count": 20}
            )
        ]
        
        output_file = os.path.join(self.temp_dir, "test_output.csv")
        self.pipeline._save_results(contracts, output_file)
        
        # Check CSV file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check JSON file was created
        json_file = output_file.replace('.csv', '.json')
        self.assertTrue(os.path.exists(json_file))
        
        # Verify CSV content
        import pandas as pd
        df = pd.read_csv(output_file)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['contract_id'], 'test_001')
        
        # Verify JSON content
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['contract_id'], 'test_001')
    
    @patch('contract_processor.ContractProcessingPipeline.run')
    def test_pipeline_error_handling(self, mock_run):
        """Test pipeline error handling"""
        mock_run.side_effect = Exception("Test error")
        
        with self.assertRaises(Exception):
            self.pipeline.run(self.temp_dir)


class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a sample text file (simulating a simple contract)
        self.sample_contract = """
        SERVICE AGREEMENT
        
        This agreement may be terminated by either party with 30 days written notice.
        
        All confidential information shall remain confidential and not be disclosed
        to third parties without written consent.
        
        The service provider shall not be liable for any indirect or consequential 
        damages arising from this agreement.
        """
        
        # Create sample PDF (we'll mock the PDF extraction)
        self.sample_file = os.path.join(self.temp_dir, "sample_contract.txt")
        with open(self.sample_file, 'w') as f:
            f.write(self.sample_contract)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('contract_processor.openai.ChatCompletion.create')
    @patch('contract_processor.DocumentProcessor.extract_text_from_pdf')
    def test_full_pipeline_mock(self, mock_pdf_extract, mock_openai):
        """Test full pipeline with mocked components"""
        # Mock PDF extraction
        mock_pdf_extract.return_value = self.sample_contract
        
        # Mock OpenAI responses
        mock_responses = [
            "Either party may terminate with 30 days notice",  # Termination
            "Confidential information must remain confidential",  # Confidentiality  
            "No liability for indirect damages",  # Liability
            "Service agreement with termination and confidentiality clauses"  # Summary
        ]
        
        def mock_openai_side_effect(*args, **kwargs):
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = mock_responses.pop(0)
            return response
        
        mock_openai.side_effect = mock_openai_side_effect
        
        # Create a fake PDF file
        pdf_file = os.path.join(self.temp_dir, "contract.pdf")
        Path(pdf_file).touch()
        
        # Run pipeline
        pipeline = ContractProcessingPipeline(api_key="test-key")
        
        with patch.object(pipeline.doc_processor, 'load_contracts') as mock_load:
            mock_contract = ContractData("test_001", "contract.pdf", self.sample_contract)
            mock_load.return_value = [mock_contract]
            
            contracts = pipeline.run(
                data_dir=self.temp_dir,
                output_file=os.path.join(self.temp_dir, "results.csv"),
                max_contracts=1,
                build_search_index=False
            )
        
        # Verify results
        self.assertEqual(len(contracts), 1)
        contract = contracts[0]
        
        self.assertIn("terminate", contract.termination_clause.lower())
        self.assertIn("confidential", contract.confidentiality_clause.lower())
        self.assertIn("liability", contract.liability_clause.lower())
        self.assertTrue(len(contract.summary) > 0)


def run_performance_tests():
    """Run performance benchmarks"""
    print("🚀 Running Performance Tests")
    print("=" * 40)
    
    import time
    import psutil
    
    # Test text normalization performance
    processor = DocumentProcessor()
    large_text = "Sample text " * 10000  # ~120KB of text
    
    start_time = time.time()
    normalized = processor.normalize_text(large_text)
    normalization_time = time.time() - start_time
    
    print(f"Text normalization: {normalization_time:.4f}s for {len(large_text)} characters")
    
    # Test memory usage
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.2f} MB")


if __name__ == '__main__':
    # Run unit tests
    print("🧪 Running Unit Tests")
    print("=" * 40)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance tests if all unit tests pass
    if result.wasSuccessful():
        print("\n" + "="*50)
        run_performance_tests()
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Please fix issues before proceeding.")
        exit(1)
