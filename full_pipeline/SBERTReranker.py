import pyterrier as pt
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util


class SBERTReranker(pt.transformer.Transformer):
    def __init__(self, model_name='all-MiniLM-L6-v2', batch_size=32, alpha=0.2):
        print(f"Loading SBERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.alpha = alpha
        
    def transform(self, topics_and_docs):
        print(f"Reranking {len(topics_and_docs)} documents with SBERT (alpha={self.alpha})")
        
        if "text" not in topics_and_docs.columns:
            print("Warning: No document text available for reranking. Using query as fallback.")
            topics_and_docs['text'] = topics_and_docs['query']
        
        topics_and_docs['bm25_score'] = topics_and_docs['score']
        
        queries = topics_and_docs["query"].unique()
        result_dfs = []
        
        for query in queries:
            print(f"Processing query: {query}")
            query_docs = topics_and_docs[topics_and_docs["query"] == query].copy()
            
            query_docs['text'] = query_docs['text'].fillna("")
            
            query_embedding = self.model.encode(query, convert_to_tensor=True, device=self.device)
            
            all_scores = []
            for i in range(0, len(query_docs), self.batch_size):
                batch = query_docs.iloc[i:i+self.batch_size]
                doc_texts = batch["text"].tolist()
                
                doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True, device=self.device)
                
                batch_scores = util.cos_sim(query_embedding.unsqueeze(0), doc_embeddings)[0]
                all_scores.extend(batch_scores.cpu().tolist())
            
            query_docs["sbert_score"] = all_scores
            
            max_bm25 = query_docs['bm25_score'].max()
            min_bm25 = query_docs['bm25_score'].min()
            if max_bm25 > min_bm25:
                query_docs['bm25_norm'] = (query_docs['bm25_score'] - min_bm25) / (max_bm25 - min_bm25)
            else:
                query_docs['bm25_norm'] = 1.0
                
            query_docs["score"] = (1 - self.alpha) * query_docs['bm25_norm'] + self.alpha * query_docs['sbert_score']
            
            query_docs = query_docs.sort_values("score", ascending=False)
            result_dfs.append(query_docs)
        
        return pd.concat(result_dfs, ignore_index=True)

class DatasetTextRetriever(pt.transformer.Transformer):
    def __init__(self, dataset):
        self.dataset = dataset
        print("Pre-loading corpus for text retrieval...")
        self.docs = {}
        for doc in self.dataset.get_corpus_iter():
            docno = doc.get('docno', '')
            if docno:
                self.docs[docno] = {
                    'abstract': doc.get('abstract', ''),
                    'title': doc.get('title', '')
                }
        print(f"Loaded {len(self.docs)} documents from corpus")
        
    def transform(self, df):
        df = df.copy()
        df['text'] = df['docno'].apply(self._get_text)
        return df
        
    def _get_text(self, docno):
        doc = self.docs.get(docno, {})
        text = doc.get('abstract', '')
        if not text:
            text = doc.get('title', '')
        return text if text else ""
