"""
Configuration file for Legal Contract Processing Pipeline
========================================================

Centralized configuration management for the contract processing pipeline.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class LLMConfig:
    """Configuration for LLM processing"""
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    max_tokens_clause: int = 500
    max_tokens_summary: int = 200
    temperature_extraction: float = 0.1
    temperature_summary: float = 0.3
    timeout_seconds: int = 30
    max_retries: int = 3
    
    def __post_init__(self):
        # Auto-load API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")

@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    max_contracts: int = 50
    max_text_length: int = 50000  # Maximum characters per contract
    min_text_length: int = 100    # Minimum characters to be valid
    chunk_size: int = 4000        # Characters per LLM request
    normalize_text: bool = True
    remove_page_numbers: bool = True
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.pdf', '.docx', '.txt']

@dataclass
class SemanticSearchConfig:
    """Configuration for semantic search"""
    enabled: bool = True
    model_name: str = "all-MiniLM-L6-v2"
    index_type: str = "flat"  # 'flat' or 'ivf'
    similarity_threshold: float = 0.5
    max_results: int = 10
    cache_embeddings: bool = True

@dataclass
class OutputConfig:
    """Configuration for output generation"""
    csv_output: bool = True
    json_output: bool = True
    include_full_text: bool = False
    include_metadata: bool = True
    output_dir: str = "./outputs"
    log_file: str = "contract_processing.log"
    log_level: str = "INFO"

@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    llm: LLMConfig = None
    processing: ProcessingConfig = None
    semantic_search: SemanticSearchConfig = None
    output: OutputConfig = None
    
    # Pipeline behavior
    parallel_processing: bool = False
    batch_size: int = 5
    save_intermediate: bool = False
    validate_results: bool = True
    
    def __post_init__(self):
        if self.llm is None:
            self.llm = LLMConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.semantic_search is None:
            self.semantic_search = SemanticSearchConfig()
        if self.output is None:
            self.output = OutputConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PipelineConfig':
        """Create configuration from dictionary"""
        config = cls()
        
        # Update LLM config
        if 'llm' in config_dict:
            llm_dict = config_dict['llm']
            config.llm = LLMConfig(**llm_dict)
        
        # Update processing config
        if 'processing' in config_dict:
            proc_dict = config_dict['processing']
            config.processing = ProcessingConfig(**proc_dict)
        
        # Update semantic search config
        if 'semantic_search' in config_dict:
            search_dict = config_dict['semantic_search']
            config.semantic_search = SemanticSearchConfig(**search_dict)
        
        # Update output config
        if 'output' in config_dict:
            output_dict = config_dict['output']
            config.output = OutputConfig(**output_dict)
        
        # Update pipeline config
        for key, value in config_dict.items():
            if key not in ['llm', 'processing', 'semantic_search', 'output']:
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from JSON file"""
        import json
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'llm': {
                'api_key': self.llm.api_key,
                'model': self.llm.model,
                'max_tokens_clause': self.llm.max_tokens_clause,
                'max_tokens_summary': self.llm.max_tokens_summary,
                'temperature_extraction': self.llm.temperature_extraction,
                'temperature_summary': self.llm.temperature_summary,
                'timeout_seconds': self.llm.timeout_seconds,
                'max_retries': self.llm.max_retries,
            },
            'processing': {
                'max_contracts': self.processing.max_contracts,
                'max_text_length': self.processing.max_text_length,
                'min_text_length': self.processing.min_text_length,
                'chunk_size': self.processing.chunk_size,
                'normalize_text': self.processing.normalize_text,
                'remove_page_numbers': self.processing.remove_page_numbers,
                'supported_formats': self.processing.supported_formats,
            },
            'semantic_search': {
                'enabled': self.semantic_search.enabled,
                'model_name': self.semantic_search.model_name,
                'index_type': self.semantic_search.index_type,
                'similarity_threshold': self.semantic_search.similarity_threshold,
                'max_results': self.semantic_search.max_results,
                'cache_embeddings': self.semantic_search.cache_embeddings,
            },
            'output': {
                'csv_output': self.output.csv_output,
                'json_output': self.output.json_output,
                'include_full_text': self.output.include_full_text,
                'include_metadata': self.output.include_metadata,
                'output_dir': self.output.output_dir,
                'log_file': self.output.log_file,
                'log_level': self.output.log_level,
            },
            'parallel_processing': self.parallel_processing,
            'batch_size': self.batch_size,
            'save_intermediate': self.save_intermediate,
            'validate_results': self.validate_results,
        }
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file"""
        import json
        
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

# Clause-specific configurations
CLAUSE_CONFIGS = {
    "termination": {
        "keywords": ["terminate", "termination", "end", "expire", "cancel"],
        "context_window": 500,
        "required_elements": ["notice", "conditions"],
    },
    "confidentiality": {
        "keywords": ["confidential", "proprietary", "non-disclosure", "trade secret"],
        "context_window": 400,
        "required_elements": ["information", "disclosure", "protection"],
    },
    "liability": {
        "keywords": ["liability", "liable", "damages", "limitation", "exclude"],
        "context_window": 600,
        "required_elements": ["damages", "limitation", "exclusion"],
    },
}

# Default prompts for clause extraction
DEFAULT_PROMPTS = {
    "termination": {
        "system": "You are an expert legal analyst specializing in contract termination clauses.",
        "instruction": """Extract the complete termination clause(s) from this contract. 
        Focus on:
        1. Conditions for termination
        2. Notice requirements
        3. Termination procedures
        4. Effects of termination
        
        If multiple termination provisions exist, combine them coherently.
        If no termination clause is found, respond with "No termination clause found"."""
    },
    "confidentiality": {
        "system": "You are an expert legal analyst specializing in confidentiality and non-disclosure provisions.",
        "instruction": """Extract the complete confidentiality clause(s) from this contract.
        Focus on:
        1. Definition of confidential information
        2. Disclosure restrictions
        3. Protection obligations
        4. Exceptions to confidentiality
        
        If multiple confidentiality provisions exist, combine them coherently.
        If no confidentiality clause is found, respond with "No confidentiality clause found"."""
    },
    "liability": {
        "system": "You are an expert legal analyst specializing in liability and indemnification clauses.",
        "instruction": """Extract the complete liability clause(s) from this contract.
        Focus on:
        1. Liability limitations
        2. Damage exclusions
        3. Indemnification provisions
        4. Insurance requirements
        
        If multiple liability provisions exist, combine them coherently.
        If no liability clause is found, respond with "No liability clause found"."""
    },
    "summary": {
        "system": "You are an expert legal analyst specializing in contract summarization.",
        "instruction": """Analyze this contract and provide a comprehensive summary in 100-150 words.
        
        Your summary must include:
        1. Purpose and type of agreement
        2. Key obligations of each party
        3. Important terms and conditions
        4. Notable risks, penalties, or limitations
        5. Duration or termination provisions
        
        Use clear, professional language suitable for business stakeholders."""
    }
}

# Environment-specific configurations
def get_production_config() -> PipelineConfig:
    """Get production-ready configuration"""
    config = PipelineConfig()
    config.llm.model = "gpt-4"
    config.llm.max_retries = 5
    config.processing.max_contracts = 100
    config.output.log_level = "WARNING"
    config.validate_results = True
    return config

def get_development_config() -> PipelineConfig:
    """Get development configuration"""
    config = PipelineConfig()
    config.llm.model = "gpt-3.5-turbo"
    config.processing.max_contracts = 10
    config.output.log_level = "DEBUG"
    config.semantic_search.enabled = True
    return config

def get_testing_config() -> PipelineConfig:
    """Get testing configuration"""
    config = PipelineConfig()
    config.llm.api_key = "test-key"
    config.processing.max_contracts = 5
    config.semantic_search.enabled = False
    config.output.log_level = "ERROR"
    return config

# Configuration validation
def validate_config(config: PipelineConfig) -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []
    
    # Validate LLM config
    if not config.llm.api_key:
        issues.append("LLM API key is required")
    
    if config.llm.max_tokens_clause <= 0:
        issues.append("max_tokens_clause must be positive")
    
    if not (0 <= config.llm.temperature_extraction <= 1):
        issues.append("temperature_extraction must be between 0 and 1")
    
    # Validate processing config
    if config.processing.max_contracts <= 0:
        issues.append("max_contracts must be positive")
    
    if config.processing.min_text_length >= config.processing.max_text_length:
        issues.append("min_text_length must be less than max_text_length")
    
    # Validate output config
    output_dir = Path(config.output.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        issues.append(f"Cannot create output directory: {e}")
    
    return issues

# Usage example
if __name__ == "__main__":
    # Create default configuration
    config = PipelineConfig()
    
    # Validate configuration
    issues = validate_config(config)
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid")
    
    # Save example configuration
    config.save_to_file("example_config.json")
    print("Example configuration saved to example_config.json")
    
    # Load and display configuration
    loaded_config = PipelineConfig.from_file("example_config.json")
    print(f"Loaded config - Model: {loaded_config.llm.model}")
    print(f"Max contracts: {loaded_config.processing.max_contracts}")
