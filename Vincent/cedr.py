import os
import pyterrier as pt
if not pt.java.started():
    pt.java.init()

from pyterrier_colbert.ranking import ColBERTFactory
from pyterrier_colbert.indexing import ColBERTIndexer

def cord19_generate():
    dataset = pt.get_dataset("irds:cord19/trec-covid")
    for doc in dataset.get_corpus_iter():
        title = doc.get('title', '')
        abstract = doc.get('abstract', '')

        # Create text field
        text = f"{title}\n{abstract}".strip()

        # Skip empty documents
        if text:
            yield {
                'docno': doc['docno'],
                'text': text
            }

dataset = pt.get_dataset("irds:cord19/trec-covid")
checkpoint="http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"

# Indexing
index_loc = "./index_path"

# 🔹 Step 1: Ensure ColBERT indexing is performed
if not os.path.exists(os.path.join(index_loc, "trec_covid", "ivfpq.faiss")):
    print("Index not found. Creating ColBERT index...")
    indexer = ColBERTIndexer(checkpoint, index_loc, "trec_covid", ids=True, chunksize=4)
    indexer.index(cord19_generate())
else:
    print("ColBERT index found. Skipping indexing.")
    indexref = pt.IndexFactory.of(index_loc)

indexref = pt.IndexFactory.of(index_loc)

pytcolbert = ColBERTFactory(checkpoint, index_loc, index_name="trec_covid", gpu=False)

bm25 = pt.terrier.Retriever(indexref, wmodel="BM25", metadata=['docno', 'title'])

pipeline = (bm25 % 100) >> pt.text.get_text(dataset, "abstract") >> pytcolbert.text_scorer()

dense_e2e = pytcolbert.end_to_end()
prf_rank = pytcolbert.prf(rerank=False)
prf_rerank = pytcolbert.prf(rerank=True)

# pipeline2 = (bm25 % 100) >> pt.text.get_text(dataset, "abstract") >> pytcolbert.prf_rank()

topic = 'description'
print(f"\nRunning experiment for topic: {topic}")

results = pt.Experiment(
    [
        bm25 % 100,
        pipeline,
        dense_e2e,
        prf_rank,
        prf_rerank
    ],
    dataset.get_topics(topic),
    dataset.get_qrels(),
    eval_metrics=["map", "recip_rank", "P.10", "ndcg_cut.10", "mrt"],
    batch_size=10,
    drop_unused=True,
    verbose=True,
    names = ["BM25", "BM25 + Normal Colbert", "ColBERT E2E","ColBERT-PRF Ranker beta=1","ColBERT-PRF ReRanker beta=1"]
)

print(results)