import os
from pathlib import Path

import numpy as np
import pyterrier as pt
from fast_forward.index import OnDiskIndex, Mode
from fast_forward.util import Indexer
from fast_forward.util.pyterrier import FFScore, FFInterpolate
from pyterrier.measures import RR, nDCG, MAP

from MiniLM.encoder import MiniLMEncoder

BASE_DIR = Path.cwd()
MODEL = "MiniLM" # TODO: generalize


# Helper function
def docs_iter():
    for d in dataset.get_corpus_iter():
        yield {
            "doc_id": d["docno"],  # Map docno → doc_id for Fast-Forward
            "text": f"{d['title']} {d.get('abstract', '')}".strip()
        }


def setup_ff_index():
    ff_index_path = BASE_DIR / "ff_indexes" / f"ffindex_cord19_{MODEL.lower()}.h5"
    if ff_index_path.exists():  # Use existing index
        ff_index = OnDiskIndex.load(
            ff_index_path,
            query_encoder=MiniLMEncoder(), # TODO: generalize
            mode=Mode.MAXP,
            # Super important that you use Mode.MAXP. Apparently this tells FFIndex to use all the text.
        )
    else:  # Create new one if it isn't already present
        ff_index = OnDiskIndex(
            ff_index_path,
            query_encoder=MiniLMEncoder(), # TODO: generalize
            mode=Mode.MAXP,
        )

        print(f"Indexing documents with {MODEL}...")
        Indexer(ff_index, MiniLMEncoder(), batch_size=8).from_dicts(docs_iter()) # TODO: Generalize

    print("Number of documents in index:", len(ff_index))
    ff_index = ff_index.to_memory()

    return FFScore(ff_index)


def tune_hyperparameter_alpha(dataset, bm25, ff_score, ff_int):
    # Hyperparameter Tuning
    return pt.GridSearch(
        bm25 % 100 >> ff_score >> ff_int,
        {ff_int: {"alpha": np.arange(0, 1, 0.1)}},
        dataset.get_topics('title'),
        dataset.get_qrels(),
        "map",
        verbose=True,
    )


def run_experiment(pipeline, topic='description'):
    # Apply MiniLM as a re-ranker
    print(f"\nRunning experiment with alpha={ff_int.alpha}... for topic: {topic}")

    # Run experiment comparing BM25 and BM25 + MiniLM
    experiment = pt.Experiment(
        [bm25 % 100, pipeline % 100],
        dataset.get_topics(topic),
        dataset.get_qrels(),
        eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100],
        names=["BM25", "BM25 + MiniLM"]
    )

    # Write output to csv
    filename = f"results/results-{MODEL}.txt"
    experiment.to_csv(filename, sep='\t', encoding='utf-8', index=False, header=True)

    # Print experiment
    print(experiment)

    return experiment


if __name__ == "__main__":
    # Initialize PyTerrier
    if not pt.started():
        pt.init()

    # Load TREC-COVID dataset
    dataset = pt.get_dataset("irds:cord19/trec-covid")

    # Indexing
    index_loc = "./index_path"
    if not os.path.exists(os.path.join(index_loc, "data.properties")):
        indexer = pt.IterDictIndexer(index_loc)
        indexref = indexer.index(dataset.get_corpus_iter(), fields=("title", "abstract"), meta=["docno", "title"])
    else:
        indexref = pt.IndexRef.of(os.path.join(index_loc, "data.properties"))

    count = sum(1 for _ in dataset.get_corpus_iter())
    print("Number of documents:", count)

    # Baseline retrieval using BM25
    bm25 = pt.BatchRetrieve(indexref, wmodel="BM25", metadata=["docno", "title"])

    # FF Indexing
    ff_score = setup_ff_index()

    # Initialize alpha as best value
    ff_int = FFInterpolate(alpha=0.2)

    # Potential hypertuning
    tune = False
    if tune:
        tuned_alpha = tune_hyperparameter_alpha(dataset, bm25, ff_score, ff_int)
        ff_int = FFInterpolate(alpha=tuned_alpha)

    pipeline = bm25 % 100 >> ff_score >> ff_int # TODO: generalize pipeline

    run_experiment(pipeline, 'title')
