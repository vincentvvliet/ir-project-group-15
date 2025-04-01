import pandas as pd
import pyterrier as pt
import subprocess
import os
if not pt.java.started():
    pt.java.init()

from pyterrier.measures import RR, nDCG, MAP
from pyterrier_dr import TctColBert, TasB

dataset = pt.get_dataset("irds:cord19/trec-covid")
index_loc = "./index_path"

indexref = pt.IndexFactory.of(index_loc)

topic = 'description'

bm25 = pt.terrier.Retriever(indexref, wmodel="BM25", metadata=['docno', 'text'])
pipeline = bm25 >> pt.text.get_text(dataset, "abstract") >> TasB.dot().text_scorer()
# pipeline = (bm25 % 100) >> pt.text.get_text(dataset, "abstract") >> RetroMAE.msmarco_finetune().text_scorer()
# pipeline = (bm25 % 100) >> pt.text.get_text(dataset, "abstract") >> GTR.base().text_scorer()
# pipeline = (bm25 % 100) >> pt.text.get_text(dataset, "abstract") >> Ance.firstp().text_scorer()
# pipeline = (bm25 % 100) >> pt.text.get_text(dataset, "abstract") >> E5.large().text_scorer()

print(f"\nRunning experiment for topic: {topic}")
results = pt.Experiment(
    [bm25 % 100, pipeline % 100],
    dataset.get_topics(topic),
    dataset.get_qrels(),
    eval_metrics=["map", "recip_rank", "P.10", "ndcg_cut.10", "mrt"],
    names=["BM25", "BM25 + Ance Base"],
    verbose=True,
)

print(results)