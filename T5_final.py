#%pip install -q python-terrier
#%pip install -q --upgrade git+https://github.com/terrierteam/pyterrier_t5.git

import pyterrier as pt
pt.init()

from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker
monoT5 = MonoT5ReRanker(text_field='abstract')
duoT5 = DuoT5ReRanker(text_field='abstract')

dataset = pt.get_dataset("irds:cord19/trec-covid")
index_loc = "./index_path"
indexer = pt.IterDictIndexer(index_loc, fields=True, text_attrs=["title", "abstract"])
indexref = indexer.index(dataset.get_corpus_iter())

count = sum(1 for _ in dataset.get_corpus_iter())
print("Number of documents:", count)

# Retrieve top 100 documents with BM25.
bm25 = pt.BatchRetrieve(indexref, wmodel="BM25") % 100

# Re-rank top 100 using monoT5.
mono_pipeline = bm25 >> pt.text.get_text(dataset, "abstract") >> monoT5 

# Only take the top 10 from mono_pipeline, then re-rank them with duoT5.
duo_pipeline = (mono_pipeline % 10) >> duoT5 

pt.Experiment(
  [
    bm25,             # Evaluates top 100 from BM25.
    mono_pipeline ,    # Evaluates re-ranking of top 100 by monoT5.
    duo_pipeline,     # Evaluates duoT5 re-ranking on only the top 10 of monoT5.
  ],
  dataset.get_topics('description'),
  dataset.get_qrels(),
  names=["BM25", "BM25 >> monoT5"],#, "BM25 >> monoT5 >> duoT5"],
  eval_metrics=["map", "recip_rank", "P.10", "ndcg_cut.10"]
)
