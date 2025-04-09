# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 01:01:00 2025

@author: maber
"""
import numpy as np
import pyterrier as pt
from pathlib import Path
import torch
from fast_forward.encoder import TCTColBERTQueryEncoder, TCTColBERTDocumentEncoder
from fast_forward.index import OnDiskIndex, Mode
from fast_forward.util import Indexer
from fast_forward.util.pyterrier import FFScore, FFInterpolate

import pandas as pd

# Base directory for all model storage
BASE_DIR = Path(r"E:\ml_models\cord10_tct_colbert_google") # Point this to the folder containing
# the ffindex_cord19_tct_colbert.h5 file
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize PyTerrier if not already done
if not pt.started():
    pt.java.init()

# Check for CUDA availability
print("CUDA available:", torch.cuda.is_available())
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the CORD19/TREC-COVID dataset
dataset = pt.get_dataset("irds:cord19/trec-covid")
# devset = pt.get_dataset("irds:cord19/trec-covid/round3") # devset for tuning parameters (optional)
# testset = pt.get_dataset("irds:cord19/trec-covid/round4") # testset to test on a different dataset (optional)

#%% Check data if you want
# dat = pd.DataFrame(dataset.get_corpus_iter())

# %% Create a lexical index for bm25
print("\nCreating lexical index...")

terrier_index_dir = BASE_DIR / "terrier_index"

if not(terrier_index_dir / 'data.properties').exists():
    indexer = pt.IterDictIndexer(
        str(terrier_index_dir),
        pt.index.IndexingType.MEMORY
        # text_attrs=["title", "abstract"],
    )
    
    # During BM25 indexing:
    index_ref = indexer.index(
        dataset.get_corpus_iter(), 
        fields=["title", "abstract"],  # Use actual available fields
        meta=["docno", "text"]       # Store docno as primary key
    )
else: 
    index_ref = pt.IndexFactory.of(str(terrier_index_dir))

#%% BM25 baseline
bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25", metadata=["docno", "text"], controls={"bm25.b" : 0.75, "bm25.k_1": 0.75, "bm25.k_3": 0.75})

#%% Optionally perform hyperparameter tuning for bm25 before adding other method to pipeline
# grid_search = pt.GridSearch(
#     bm25,
#     {bm25: {"bm25.b"  : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 ],
#             "bm25.k_1": [0.3, 0.6, 0.9, 1.2, 1.4, 1.6, 2],
#             "bm25.k_3": [0.5, 2, 4, 6, 8, 10, 12, 14, 20]
#     }},
#     dataset.get_topics('description'),
#     dataset.get_qrels(),
#     "map",
#     verbose=True,
# )

# # Evaluate baseline
# print("\nBM25 Baseline:")
# pt.Experiment(
#     [bm25],
#     dataset.get_topics('description'),
#     dataset.get_qrels(),
#     eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100],
# ).to_csv(str(BASE_DIR / "results_bm25.csv"))



#%%
# Initialize TCT-ColBERT encoders (can change these for other encoders I think)
print("\nInitializing TCT-ColBERT encoders...")
q_encoder = TCTColBERTQueryEncoder(
    model='castorini/tct_colbert-msmarco',
    device=device,
    max_length=36 # max query length
)

d_encoder = TCTColBERTDocumentEncoder(
    model='castorini/tct_colbert-msmarco',
    device=device,
    max_length=512 # max length on documents
)

# Create Fast-Forward index path
ff_index_path = BASE_DIR / "ffindex_cord19_tct_colbert.h5"
def docs_iter():
    for d in dataset.get_corpus_iter():
        yield {
            "doc_id": d["docno"],  # Map docno → doc_id for Fast-Forward
            "text": f"{d['title']} {d.get('abstract', '')}".strip()
        }

if ff_index_path.exists(): # Use existing index
    ff_index = OnDiskIndex.load(
        ff_index_path,
        query_encoder=q_encoder,
        mode=Mode.MAXP, # Super important that you use Mode.MAXP. Apparently this tells FFIndex to use all the text.
    )
else: # Create new one if it isn't already present
    ff_index = OnDiskIndex(
        ff_index_path,
        query_encoder=q_encoder,
        mode=Mode.MAXP,
    )
    
    print("Indexing documents with TCT-ColBERT...")
    Indexer(ff_index, d_encoder, batch_size=8).from_dicts(docs_iter())

ff_index = ff_index.to_memory()
 
#%% Defining FF index reranker using tct_colbert
ff_score = FFScore(ff_index)
ff_int = FFInterpolate(alpha=0.5) # initial alpha guess
# %% Hyperparameter tuning

tune=False
if tune:
    print("\nTuning parameters")
    
    grid_search = pt.GridSearch(
        bm25 % 100 >> ff_score >> ff_int,
        {ff_int: {"alpha": np.arange(0,1,0.1)},
         # bm25: {
         #        "bm25.b"  : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 ],
         #        "bm25.k_1": [0.3, 0.6, 0.9, 1.2, 1.4, 1.6, 2],
         #        "bm25.k_3": [0.5, 2, 4, 6, 8, 10, 12, 14, 20]}
        }, # Can tune BM25 alongside it, but you need a GPU to do it in reasonable time.
        dataset.get_topics('description'), # using 'description' as the topic
        dataset.get_qrels(),
        "map",
        verbose=True,
    )
    # grid_search.to_csv(str(BASE_DIR / "grid_search_results.csv")) # Optionally save grid search results

# Current found best score: 0.088
# grid_scan = pt.GridScan(
#     bm25 % 100 >> ff_score >> ff_int,
#     {ff_int: {"alpha": np.arange(0,1,0.01)}},
#     devset.get_topics('description'),
#     devset.get_qrels(),
#     "map",
#     verbose=True,
# ) # didn't figure out how to use GridScan, I think it's not necessary but i'll leave it in


# %%Run final experiment on test set with best alpha
from pyterrier.measures import RR, nDCG, MAP, P, R

ff_int = FFInterpolate(alpha=0.2)

pipeline = bm25 % 100 >> ff_score >> ff_int
print(f"\nRunning experiment with alpha={ff_int.alpha}...")
experiment = pt.Experiment(
    [bm25 % 100, pipeline % 100],
    dataset.get_topics('description'),
    dataset.get_qrels(),
    eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100, 'mrt', R @ 10, R @ 25, R @ 50],
    # eval_metrics=["map", "recip_rank", "P.10", "ndcg_cut.10"],
    names=["BM25", "BM25 >> TCT-ColBERT"],
)
experiment.to_csv(str(BASE_DIR / "final_results.csv"))
print(experiment)
