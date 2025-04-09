# Evaluating Common Methodologies with Varying Pre-trained Models for Reranking of TREC-COVID (Group 15)
Using BM25 (baseline), TAS-B, SBERT, ColBERT, and T5 to gauge which model and methodology for neural network reranking is best for 'off-the-shelf' applications on specialised datasets such as TREC-COVID.


# Repository Structure
All implementations and experiments were originally created seperately, hence each group member has a folder associated with their models and method.

The *full_pipeline* directory holds the main file called *full_pipeline.py* which, when executed, should generate the index and run all experiments for all query types on CPU or GPU (```torch.cuda.is_available()```). 

**Warning**: Fully executing full_pipeline.py will download 6GB worth of data and model weights.


# Python 3.9 Dependencies
To install all dependencies on a new venv run ```pip install -r requirements.txt``` at the root of the repository.

Some packages are automatically installed since they are dependencies of others, but for the sake of clarity all key packages are listed.

## Machine Learning & AI
- **torch** (2.6.0+cu124) - PyTorch with CUDA 12.4 support
- **transformers** (4.50.3) - Hugging Face Transformers library
- **sentence-transformers** (4.0.1) - For creating document/text embeddings
- **scikit-learn** (1.6.1) - General ML algorithms

## IR (Information Retrieval) Tools
- **PyTerrier** (0.13.0) - IR platform (requires Java)
- **pyterrier-colbert** (0.0.1) - ColBERT integration for PyTerrier
- **pyterrier-t5** (0.1.0) - T5 models integration
- **pyterrier-dr** (0.6.1) - Dense retrieval for PyTerrier
- **ir_datasets** (0.5.10) - Standard IR datasets
- **ir_measures** (0.3.7) - Evaluation metrics for IR

## Data Processing
- **numpy** (1.26.3) - Numerical computing
- **pandas** (1.5.3) - Data manipulation and analysis
- **scipy** (1.13.1) - Scientific computing
- **pyarrow** (19.0.1) - Efficient data serialization

## Visualization
- **matplotlib** (3.9.4) - Plotting library

## Web & API
- **Flask** (2.3.3) - Web framework
- **requests** (2.32.3) - HTTP library
- **beautifulsoup4** (4.13.3) - Web scraping

## MLOps & Experiment Tracking
- **mlflow** (1.30.1) - ML lifecycle management

## Utilities
- **tqdm** (4.67.1) - Progress bars
- **joblib** (1.4.2) - Parallel computing
- **cloudpickle** (2.2.1) - Extended pickling support

## Note
Some packages require additional system dependencies:
- **PyTerrier** and related packages require Java
- **torch** with CUDA requires compatible NVIDIA drivers

