"""
Example Usage of Legal Contract Processing Pipeline
=================================================

This file demonstrates various ways to use the contract processing pipeline.
"""

import os
from pathlib import Path
from contract_processor import ContractProcessingPipeline, ContractData

# Set your OpenAI API key
API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

def basic_example():
    """Basic usage example"""
    print("🚀 Basic Pipeline Example")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = ContractProcessingPipeline(api_key=API_KEY)
    
    # Process contracts
    contracts = pipeline.run(
        data_dir="./sample_contracts",
        output_file="basic_analysis.csv",
        max_contracts=10
    )
    
    print(f"✅ Processed {len(contracts)} contracts")
    
    # Display sample results
    for contract in contracts[:2]:
        print(f"\n📋 {contract.contract_id}:")
        print(f"Summary: {contract.summary[:100]}...")
        print(f"Termination: {contract.termination_clause[:100]}...")

def advanced_example():
    """Advanced usage with semantic search"""
    print("\n🔍 Advanced Pipeline with Semantic Search")
    print("=" * 50)
    
    pipeline = ContractProcessingPipeline(api_key=API_KEY, model="gpt-4")
    
    # Process contracts
    contracts = pipeline.run(
        data_dir="./sample_contracts",
        output_file="advanced_analysis.csv",
        max_contracts=25,
        build_search_index=True
    )
    
    # Demonstrate semantic search
    if contracts:
        print("\n🔎 Semantic Search Demo:")
        
        search_queries = [
            "contract termination notice requirements",
            "confidential information protection",
            "liability limitations and damages",
            "intellectual property rights",
            "payment terms and conditions"
        ]
        
        for query in search_queries:
            print(f"\nQuery: '{query}'")
            results = pipeline.search_clauses(query, k=3)
            
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                score = result['similarity_score']
                clause = result['clause'][:150] + "..."
                
                print(f"  {i}. {metadata['contract_id']} ({metadata['clause_type']}) - Score: {score:.3f}")
                print(f"     {clause}")

def custom_processing_example():
    """Example of custom contract processing"""
    print("\n⚙️ Custom Processing Example")
    print("=" * 50)
    
    pipeline = ContractProcessingPipeline(api_key=API_KEY)
    
    # Load contracts manually
    contracts = pipeline.doc_processor.load_contracts("./sample_contracts", max_contracts=5)
    
    # Process each contract with custom logic
    processed_contracts = []
    
    for contract in contracts:
        print(f"Processing {contract.contract_id}...")
        
        # Add custom metadata
        contract.extraction_metadata = {
            "custom_field": "example_value",
            "processing_version": "2.0"
        }
        
        # Process with LLM
        processed_contract = pipeline.llm_processor.process_contract(contract)
        
        # Custom post-processing
        if "termination" in processed_contract.termination_clause.lower():
            processed_contract.extraction_metadata["has_termination"] = True
        
        processed_contracts.append(processed_contract)
    
    # Save with custom filename
    pipeline._save_results(processed_contracts, "custom_analysis.csv")
    
    print(f"✅ Custom processing completed for {len(processed_contracts)} contracts")

def error_handling_example():
    """Example demonstrating error handling"""
    print("\n🛡️ Error Handling Example")
    print("=" * 50)
    
    pipeline = ContractProcessingPipeline(api_key="invalid-key")  # Intentionally invalid
    
    try:
        # This will fail due to invalid API key
        contracts = pipeline.run(
            data_dir="./sample_contracts",
            output_file="error_test.csv",
            max_contracts=2
        )
    except Exception as e:
        print(f"❌ Expected error occurred: {e}")
        print("💡 This demonstrates the pipeline's error handling capabilities")
    
    # Show how to handle missing data directory
    try:
        pipeline_valid = ContractProcessingPipeline(api_key=API_KEY)
        contracts = pipeline_valid.run(
            data_dir="./nonexistent_directory",
            output_file="missing_dir_test.csv"
        )
    except FileNotFoundError as e:
        print(f"❌ Directory not found error: {e}")
        print("💡 Make sure your data directory exists")

def performance_monitoring_example():
    """Example showing performance monitoring"""
    print("\n📊 Performance Monitoring Example")
    print("=" * 50)
    
    import time
    import psutil
    
    # Monitor memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    start_time = time.time()
    
    pipeline = ContractProcessingPipeline(api_key=API_KEY)
    contracts = pipeline.run(
        data_dir="./sample_contracts",
        output_file="performance_test.csv",
        max_contracts=10
    )
    
    end_time = time.time()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Performance metrics
    processing_time = end_time - start_time
    contracts_per_minute = (len(contracts) / processing_time) * 60
    memory_usage = final_memory - initial_memory
    
    print(f"\n📈 Performance Metrics:")
    print(f"Total time: {processing_time:.2f} seconds")
    print(f"Contracts processed: {len(contracts)}")
    print(f"Processing rate: {contracts_per_minute:.2f} contracts/minute")
    print(f"Memory usage: {memory_usage:.2f} MB")
    print(f"Average time per contract: {processing_time/len(contracts):.2f} seconds")

def batch_processing_example():
    """Example of processing contracts in batches"""
    print("\n🔄 Batch Processing Example")
    print("=" * 50)
    
    pipeline = ContractProcessingPipeline(api_key=API_KEY)
    
    # Get all contract files
    contract_files = list(Path("./sample_contracts").glob("*.pdf"))
    batch_size = 5
    
    all_processed_contracts = []
    
    # Process in batches
    for i in range(0, len(contract_files), batch_size):
        batch_files = contract_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}: {len(batch_files)} files")
        
        # Create temporary directory for batch
        batch_dir = f"./temp_batch_{i//batch_size + 1}"
        os.makedirs(batch_dir, exist_ok=True)
        
        # Copy files to batch directory (in practice, you'd process directly)
        # For this example, we'll just process from the original directory
        
        try:
            contracts = pipeline.run(
                data_dir="./sample_contracts",
                output_file=f"batch_{i//batch_size + 1}_analysis.csv",
                max_contracts=batch_size
            )
            all_processed_contracts.extend(contracts)
            
        except Exception as e:
            print(f"❌ Batch {i//batch_size + 1} failed: {e}")
            continue
    
    print(f"✅ Batch processing completed: {len(all_processed_contracts)} total contracts")

def main():
    """Run all examples"""
    
    # Check if sample contracts directory exists
    if not Path("./sample_contracts").exists():
        print("⚠️ Sample contracts directory not found.")
        print("Please create './sample_contracts' directory and add some PDF/DOCX contract files.")
        print("You can download sample contracts from the CUAD dataset:")
        print("https://www.atticusprojectai.org/cuad")
        return
    
    # Check API key
    if API_KEY == "your-api-key-here":
        print("⚠️ Please set your OpenAI API key in the API_KEY variable or OPENAI_API_KEY environment variable")
        return
    
    print("🎯 Legal Contract Processing Pipeline - Examples")
    print("=" * 60)
    
    # Run examples (comment out any you don't want to run)
    try:
        basic_example()
        advanced_example()
        custom_processing_example()
        performance_monitoring_example()
        batch_processing_example()
        
        # Skip error handling example by default
        # error_handling_example()
        
    except KeyboardInterrupt:
        print("\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"❌ Example execution failed: {e}")
        raise
    
    print("\n🎉 All examples completed successfully!")
    print("\nGenerated files:")
    for file in ["basic_analysis.csv", "advanced_analysis.csv", "custom_analysis.csv", "performance_test.csv"]:
        if Path(file).exists():
            print(f"  ✅ {file}")

if __name__ == "__main__":
    main()
