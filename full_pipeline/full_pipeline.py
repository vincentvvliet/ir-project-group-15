# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 17:21:31 2025

@author: maber
"""

import pyterrier as pt

if not pt.java.started():
    pt.java.init()
from SBERTReranker import SBERTReranker, DatasetTextRetriever
# pyterrier colbert import
from pyterrier_colbert.ranking import ColBERTFactory

# pyterrier T5 import
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker

# sentence transformer import
from sentence_transformers import SentenceTransformer, util

# pyterrier dr import
from pyterrier_dr import TasB

# pyterrier metrics
from pyterrier.measures import RR, nDCG, MAP, P, R
# %% Retrieve trec-covid dataset:

dataset = pt.get_dataset("irds:cord19/trec-covid")
# data= pd.DataFrame(dataset.get_corpus_iter())

doc_fetcher = DatasetTextRetriever(dataset)

# %% Define premade index path (replace path with your own)
index_loc = r"E:\ml_models\indexes\cord19_trec_index_newindex"
indexref = pt.IndexFactory.of(index_loc)

bm25 = pt.terrier.Retriever(indexref, wmodel="BM25", metadata=['docno', 'text'])

#%% Colbert implementation
colbert_factory = ColBERTFactory(
    colbert_model="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip",
    index_root=index_loc,
    index_name="trec_covid",
    gpu=False,  # change this if, like me, you don't have an Nvidia GPU :)
    )

colbert_pipeline = (bm25 % 100) >> doc_fetcher >> colbert_factory.text_scorer()

#%% Minilm implementation

sbert_reranker = SBERTReranker(model_name='all-MiniLM-L6-v2', alpha=0.2) # set it to minilm instead of bert

minilm_pipeline = (bm25 % 100) >> doc_fetcher >> sbert_reranker

# %% T5 implementation

monoT5 = MonoT5ReRanker(text_field='text')
# duoT5 = DuoT5ReRanker(text_field='abstract')

# Re-rank top 100 using monoT5.
monoT5_pipeline = (bm25 % 100) >> doc_fetcher >> monoT5 


#%% TasB implementation

tasB_pipeline = (bm25 % 100) >> doc_fetcher >> TasB.dot().text_scorer()

#%% -------------------------------------- TESTS ---------------------------------------------------
# Title

experiment1 = pt.Experiment(
    [bm25 % 100, colbert_pipeline % 100, minilm_pipeline % 100, monoT5_pipeline % 100],
    dataset.get_topics('title'),
    dataset.get_qrels(),
    eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100, 'mrt', R @ 10, R @ 25, R @ 50],
    names=["BM25", "BM25 >> TCT-ColBERT % 100", "BM25 >> MINILM % 100", "BM25 >> MonoT5 % 100"],
)

output_file = "trec_covid_results_title.run"
pt.io.write_results(experiment1, output_file)
print(f"\nResults saved to {output_file}")

#%% Description

experiment2 = pt.Experiment(
    [bm25 % 100, colbert_pipeline % 100, minilm_pipeline % 100, monoT5_pipeline % 100],
    dataset.get_topics('description'),
    dataset.get_qrels(),
    eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100, 'mrt', R @ 10, R @ 25, R @ 50],
    names=["BM25", "BM25 >> TCT-ColBERT % 100", "BM25 >> MINILM % 100", "BM25 >> MonoT5 % 100"],
)

output_file = "trec_covid_results_description.run"
pt.io.write_results(experiment2, output_file)
print(f"\nResults saved to {output_file}")

#%% Narrative

experiment3 = pt.Experiment(
    [bm25 % 100, colbert_pipeline % 100, minilm_pipeline % 100, monoT5_pipeline % 100],
    dataset.get_topics('narrative'),
    dataset.get_qrels(),
    eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100, 'mrt', R @ 10, R @ 25, R @ 50],
    names=["BM25", "BM25 >> TCT-ColBERT % 100", "BM25 >> MINILM % 100", "BM25 >> MonoT5 % 100"],
)

output_file = "trec_covid_results_narrative.run"
pt.io.write_results(experiment3, output_file)
print(f"\nResults saved to {output_file}")
