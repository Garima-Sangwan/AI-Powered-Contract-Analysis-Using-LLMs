"""
Legal Contract Processing Pipeline with LLMs
=============================================

A comprehensive solution for extracting key clauses and generating summaries 
from legal contracts using Large Language Models.

Author: AI Developer
Date: August 2025
"""

import os
import json
import csv
import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import argparse

# Core libraries
import pandas as pd
import numpy as np
from tqdm import tqdm

# Document processing
import PyPDF2
from PyPDF2 import PdfReader
import fitz  # PyMuPDF - better PDF extraction
from docx import Document

# LLM and embeddings
import openai
from sentence_transformers import SentenceTransformer
import faiss

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


@dataclass
class ContractData:
    """Data structure for contract information"""
    contract_id: str
    filename: str
    text: str
    summary: str = ""
    termination_clause: str = ""
    confidentiality_clause: str = ""
    liability_clause: str = ""
    extraction_metadata: Dict = None


class DocumentProcessor:
    """Handles document loading and text extraction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF (more robust than PyPDF2)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            self.logger.warning(f"PyMuPDF failed for {pdf_path}: {e}")
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                return text
            except Exception as e2:
                self.logger.error(f"Both PDF extractors failed for {pdf_path}: {e2}")
                return ""
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX files"""
        try:
            doc = Document(docx_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            self.logger.error(f"Failed to extract text from {docx_path}: {e}")
            return ""
    
    def normalize_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        
        # Remove page numbers and common headers/footers
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def load_contracts(self, data_dir: str, max_contracts: int = 50) -> List[ContractData]:
        """Load and process contracts from directory"""
        contracts = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory {data_dir} not found")
        
        # Get all PDF and DOCX files
        pdf_files = list(data_path.glob("*.pdf"))
        docx_files = list(data_path.glob("*.docx"))
        all_files = pdf_files + docx_files
        
        self.logger.info(f"Found {len(all_files)} contract files")
        
        for i, file_path in enumerate(tqdm(all_files[:max_contracts], desc="Loading contracts")):
            contract_id = f"contract_{i+1:03d}"
            
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                raw_text = self.extract_text_from_pdf(str(file_path))
            elif file_path.suffix.lower() == '.docx':
                raw_text = self.extract_text_from_docx(str(file_path))
            else:
                continue
            
            if not raw_text.strip():
                self.logger.warning(f"No text extracted from {file_path}")
                continue
            
            # Normalize text
            clean_text = self.normalize_text(raw_text)
            
            if len(clean_text) < 100:  # Skip very short documents
                self.logger.warning(f"Document {file_path} too short, skipping")
                continue
            
            contracts.append(ContractData(
                contract_id=contract_id,
                filename=file_path.name,
                text=clean_text
            ))
        
        self.logger.info(f"Successfully loaded {len(contracts)} contracts")
        return contracts


class LLMProcessor:
    """Handles LLM-based information extraction and summarization"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.model = model
        if api_key:
            openai.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Few-shot examples for better extraction
        self.few_shot_examples = self._load_few_shot_examples()
    
    def _load_few_shot_examples(self) -> Dict[str, str]:
        """Load few-shot examples for clause extraction"""
        return {
            "termination": """
Example termination clause:
"This Agreement may be terminated by either party upon thirty (30) days written notice to the other party. In the event of material breach, the non-breaching party may terminate immediately upon written notice."
""",
            "confidentiality": """
Example confidentiality clause:
"Each party agrees to maintain in confidence all Confidential Information received from the other party and shall not disclose such information to third parties without prior written consent."
""",
            "liability": """
Example liability clause:
"IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL OR PUNITIVE DAMAGES, EVEN IF SUCH PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES."
"""
        }
    
    def _create_clause_extraction_prompt(self, text: str, clause_type: str) -> str:
        """Create optimized prompt for clause extraction"""
        few_shot = self.few_shot_examples.get(clause_type, "")
        
        prompt = f"""You are a legal document analyst. Extract {clause_type} clauses from the following contract.

{few_shot}

Instructions:
1. Identify and extract the complete {clause_type} clause(s) from the contract
2. If multiple clauses exist, combine them coherently
3. If no {clause_type} clause exists, respond with "No {clause_type} clause found"
4. Maintain the original legal language and structure
5. Focus on the core obligations and conditions

Contract text:
{text[:4000]}  # Limit text to avoid token limits

{clause_type.title()} clause:"""
        
        return prompt
    
    def _create_summary_prompt(self, text: str) -> str:
        """Create prompt for contract summarization"""
        prompt = f"""Analyze this legal contract and provide a comprehensive summary in 100-150 words.

Focus on:
1. Purpose and nature of the agreement
2. Key obligations of each party
3. Notable risks, penalties, or limitations
4. Important terms and conditions

Contract text:
{text[:4000]}

Summary:"""
        
        return prompt
    
    def extract_clause(self, text: str, clause_type: str) -> str:
        """Extract specific clause type using LLM"""
        try:
            prompt = self._create_clause_extraction_prompt(text, clause_type)
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert legal document analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1  # Low temperature for consistent extraction
            )
            
            clause = response.choices[0].message.content.strip()
            return clause
            
        except Exception as e:
            self.logger.error(f"Error extracting {clause_type} clause: {e}")
            return f"Error extracting {clause_type} clause"
    
    def generate_summary(self, text: str) -> str:
        """Generate contract summary using LLM"""
        try:
            prompt = self._create_summary_prompt(text)
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert legal analyst specializing in contract summarization."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3  # Slightly higher for more natural summaries
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return "Error generating summary"
    
    def process_contract(self, contract: ContractData) -> ContractData:
        """Process a single contract for all extractions"""
        self.logger.info(f"Processing {contract.contract_id}")
        
        # Extract clauses
        contract.termination_clause = self.extract_clause(contract.text, "termination")
        contract.confidentiality_clause = self.extract_clause(contract.text, "confidentiality")
        contract.liability_clause = self.extract_clause(contract.text, "liability")
        
        # Generate summary
        contract.summary = self.generate_summary(contract.text)
        
        # Add metadata
        contract.extraction_metadata = {
            "text_length": len(contract.text),
            "word_count": len(contract.text.split()),
            "processed_at": pd.Timestamp.now().isoformat()
        }
        
        return contract


class SemanticSearch:
    """Bonus: Semantic search over extracted clauses using embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.clauses = []
        self.clause_metadata = []
        self.logger = logging.getLogger(__name__)
    
    def build_index(self, contracts: List[ContractData]):
        """Build FAISS index for semantic search"""
        all_clauses = []
        metadata = []
        
        for contract in contracts:
            clauses = {
                "termination": contract.termination_clause,
                "confidentiality": contract.confidentiality_clause,
                "liability": contract.liability_clause
            }
            
            for clause_type, clause_text in clauses.items():
                if clause_text and not clause_text.startswith("No ") and not clause_text.startswith("Error"):
                    all_clauses.append(clause_text)
                    metadata.append({
                        "contract_id": contract.contract_id,
                        "clause_type": clause_type,
                        "filename": contract.filename
                    })
        
        if not all_clauses:
            self.logger.warning("No valid clauses found for indexing")
            return
        
        # Generate embeddings
        self.logger.info(f"Generating embeddings for {len(all_clauses)} clauses")
        embeddings = self.embedding_model.encode(all_clauses)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.clauses = all_clauses
        self.clause_metadata = metadata
        
        self.logger.info(f"Built semantic search index with {len(all_clauses)} clauses")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar clauses"""
        if not self.index:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "clause": self.clauses[idx],
                "metadata": self.clause_metadata[idx],
                "similarity_score": float(score)
            })
        
        return results


class ContractProcessingPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.doc_processor = DocumentProcessor()
        self.llm_processor = LLMProcessor(api_key, model)
        self.semantic_search = SemanticSearch()
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('contract_processing.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def run(self, data_dir: str, output_file: str = "contract_analysis.csv", 
            max_contracts: int = 50, build_search_index: bool = True) -> List[ContractData]:
        """Run the complete pipeline"""
        
        self.logger.info("Starting contract processing pipeline")
        
        # Step 1: Load and preprocess documents
        self.logger.info("Step 1: Loading contracts")
        contracts = self.doc_processor.load_contracts(data_dir, max_contracts)
        
        if not contracts:
            raise ValueError("No contracts loaded successfully")
        
        # Step 2: Process contracts with LLM
        self.logger.info("Step 2: Processing contracts with LLM")
        processed_contracts = []
        
        for contract in tqdm(contracts, desc="Processing contracts"):
            try:
                processed_contract = self.llm_processor.process_contract(contract)
                processed_contracts.append(processed_contract)
            except Exception as e:
                self.logger.error(f"Failed to process {contract.contract_id}: {e}")
                continue
        
        # Step 3: Save results
        self.logger.info("Step 3: Saving results")
        self._save_results(processed_contracts, output_file)
        
        # Step 4: Build semantic search index (bonus)
        if build_search_index and processed_contracts:
            self.logger.info("Step 4: Building semantic search index")
            self.semantic_search.build_index(processed_contracts)
        
        self.logger.info(f"Pipeline completed. Processed {len(processed_contracts)} contracts")
        return processed_contracts
    
    def _save_results(self, contracts: List[ContractData], output_file: str):
        """Save results to CSV and JSON"""
        
        # Prepare data for CSV
        csv_data = []
        json_data = []
        
        for contract in contracts:
            row = {
                "contract_id": contract.contract_id,
                "filename": contract.filename,
                "summary": contract.summary,
                "termination_clause": contract.termination_clause,
                "confidentiality_clause": contract.confidentiality_clause,
                "liability_clause": contract.liability_clause,
                "text_length": contract.extraction_metadata.get("text_length", 0),
                "word_count": contract.extraction_metadata.get("word_count", 0)
            }
            csv_data.append(row)
            
            # More detailed JSON output
            json_row = row.copy()
            json_row["full_text"] = contract.text[:500] + "..." if len(contract.text) > 500 else contract.text
            json_row["metadata"] = contract.extraction_metadata
            json_data.append(json_row)
        
        # Save CSV
        csv_file = output_file
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if csv_data:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
        
        # Save JSON
        json_file = output_file.replace('.csv', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {csv_file} and {json_file}")
    
    def search_clauses(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar clauses using semantic search"""
        return self.semantic_search.search(query, k)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Legal Contract Processing Pipeline")
    parser.add_argument("--data_dir", required=True, help="Directory containing contract files")
    parser.add_argument("--output", default="contract_analysis.csv", help="Output file name")
    parser.add_argument("--max_contracts", type=int, default=50, help="Maximum number of contracts to process")
    parser.add_argument("--api_key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM model to use")
    parser.add_argument("--no_search", action="store_true", help="Skip building semantic search index")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ContractProcessingPipeline(args.api_key, args.model)
    
    try:
        # Run pipeline
        contracts = pipeline.run(
            data_dir=args.data_dir,
            output_file=args.output,
            max_contracts=args.max_contracts,
            build_search_index=not args.no_search
        )
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Processed {len(contracts)} contracts")
        print(f"💾 Results saved to {args.output}")
        
        # Demo semantic search if enabled
        if not args.no_search and contracts:
            print("\n🔍 Semantic search demo:")
            demo_queries = [
                "contract termination notice period",
                "confidential information disclosure",
                "limitation of liability damages"
            ]
            
            for query in demo_queries:
                results = pipeline.search_clauses(query, k=2)
                print(f"\nQuery: '{query}'")
                for i, result in enumerate(results[:2], 1):
                    print(f"  {i}. {result['metadata']['contract_id']} ({result['metadata']['clause_type']}) - Score: {result['similarity_score']:.3f}")
                    print(f"     {result['clause'][:100]}...")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
