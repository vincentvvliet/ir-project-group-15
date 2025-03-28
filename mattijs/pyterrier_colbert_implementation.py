# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 19:37:10 2025

@author: maber
"""

import time

import shutil
import os
import pandas as pd
from tqdm import tqdm

import pyterrier as pt

#%%
if not pt.java.started():
    pt.java.init()

from pyterrier.measures import RR, nDCG, MAP
from pyterrier_colbert.ranking import ColBERTFactory

# import pyterrier_colbert
# pt.java.init()

# %% Retrieve trec-covid dataset:


dataset = pt.get_dataset("irds:cord19/trec-covid")


# # %% Define premade index path (replace path with your own)
index_loc = r"E:\ml_models\indexes\cord19_trec_index_newindex"


def cord19_generate():
    dataset = pt.get_dataset("irds:cord19/trec-covid")
    for doc in dataset.get_corpus_iter():
        yield {
            'docno': doc['docno'],
            'text': f"{doc.get('title', '')}\n{doc.get('abstract', '')}".strip()
        }


# Create the indexer with appropriate field lengths
# iter_indexer = pt.IterDictIndexer(
#     index_loc,
#     meta={'docno': 20, 'text': 4096},  # docno up to 20 chars, text up to 4096 chars
#     fields=[('text', 1.0)]  # Index the text field with weight 1.0
# )
# indexref = iter_indexer.index(cord19_generate())


# indexer = pt.IterDictIndexer(index_loc, fields=True, text_attrs=["title", "abstract"])
# indexref = indexer.index(dataset.get_corpus_iter())

# colbert_index_loc = r"E:\ml_models\colbert_index"  # Separate directory
# os.makedirs(colbert_index_loc, exist_ok=True)


indexref = pt.IndexFactory.of(index_loc)

colbert_factory = ColBERTFactory(
    colbert_model="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip",
    index_root=index_loc,
    index_name="trec_covid",
    gpu=False,  # change this if, like me, you don't have an Nvidia GPU :)
    # nthreads=8
    )


#%%

data= pd.DataFrame(dataset.get_corpus_iter())
#%%

bm25 = pt.terrier.Retriever(indexref, wmodel="BM25", metadata=['docno', 'text'])
# pipeline = (bm25%100) >> pt.text.get_text(dataset, "abstract") >> colbert_factory.text_scorer()
pipeline = (bm25%100) >> pt.text.get_text(dataset, "abstract") >> colbert_factory.text_scorer()
# Or pre-filter low BM25 scores0
# pipeline = bm25 % 100 >> pt.apply.doc_score_filter(50) >> colbert_factory.text_scorer()
# %%6. Run experiment
results = pt.Experiment(
    [bm25 % 100, pipeline % 100],
    dataset.get_topics('description'),
    dataset.get_qrels(),
    eval_metrics=["map", "recip_rank", "P.10", "ndcg_cut.10"],
    names=["BM25", "BM25+ColBERT"],
    verbose=True,
)









# # Create an index (if you haven't already)
# indexer = pt.IterDictIndexer(
#     index_loc,  # Save to your preferred location
#     meta={"docno": 100, "text": 4096},  # Adjust field lengths as needed
#     overwrite=False # Set to False after first run
# )
# index_ref = indexer.index(dataset.get_corpus_iter(), fields=["text"])

# # Define BM25 retrieval
# bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", num_results=1000)  # Get top 1000 docs

# %% Retrieve index if you don't already have it (uncomment lines)
# index_loc = r"E:\ml_models\indexes\cord19_trec_index"
# Create an index (if you haven't already)
# indexer = pt.IterDictIndexer(
#     index_loc,
#     fields=True,# Adjust field lengths as needed
#     text_attrs=["title", "abstract"])

# index_ref = indexer.index(dataset.get_corpus_iter())





# # Create optimized document generator


# def trec_covid_docs():
#     dataset = pt.get_dataset("irds:cord19/trec-covid")
#     for doc in dataset.get_corpus_iter():
#         yield {
#             'docno': doc['docno'],
#             'text': f"{doc.get('title', '')}\n{doc.get('abstract', '')}",
#             'title': doc.get('title', ''),
#             'abstract': doc.get('abstract', '')
#         }


# # # 3. Build index with optimized settings
# indexer = pt.IterDictIndexer(
#     index_loc,
#     meta={'docno': 26, 'text': 2048},  # Field lengths
#     blocks=True,  # Enable block indexing for efficiency
#     verbose=True,
#     # threads=8
# )

# # indexref = indexer.index(
# #     trec_covid_docs(),
# #     fields={'text': 1.0}  # Field weights
# # )


# %% Define index for use
# index = pt.IndexFactory.of(index_loc)


# %% For some reason you need to specifically initialise java before using ColBERT

# %% Define colbert from the pyterrier_colbert package

# colbert_index_loc = r"E:\ml_models\colbert_index"  # Separate directory
# os.makedirs(colbert_index_loc, exist_ok=True)


# colbert_factory = ColBERTFactory(
#     colbert_model="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip",
#     index_root=colbert_index_loc,
#     index_name="trec_covid",
#     gpu=False,  # change this if, like me, you don't have an Nvidia GPU :)
#     # nthreads=8
# )

# docs_df = pd.DataFrame(list(trec_covid_docs()))

# Index all documents through the scorer
# _ = colbert_factory.text_scorer().index(docs_df)

# %% Define the pipeline
# bm25 = pt.terrier.Retriever(index, 'terrier_stemmed_text',  wmodel="BM25", metadata=['docno', 'text'])
# pipeline = bm25 >> colbert_factory.text_scorer()

# bm25 = pt.terrier.Retriever.from_dataset(dataset, 'terrier_stemmed_text', wmodel="BM25", metadata=['docno', 'text'])
# pipeline = bm25 >> colbert_factory.rerank()
# # %%6. Run experiment
# results = pt.Experiment(
#     [bm25 % 100, pipeline % 100],
#     dataset.get_topics(),
#     dataset.get_qrels(),
#     eval_metrics=[RR @ 10, nDCG @ 20, MAP],
#     names=["BM25", "BM25+ColBERT"]
# )
