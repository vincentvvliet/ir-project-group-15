# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 17:21:31 2025

@author: maber
"""
import json
import numpy as np
import matplotlib.pyplot as plt

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

import os

# %% Retrieve trec-covid dataset:

dataset = pt.get_dataset("irds:cord19/trec-covid")
# data= pd.DataFrame(dataset.get_corpus_iter())

doc_fetcher = DatasetTextRetriever(dataset)

index_loc = os.path.join(os.getcwd(), "index")
os.makedirs(index_loc, exist_ok=True)

def prepare_docs(corpus_iter):
    for doc in iter(corpus_iter):
        text_content = []
        if 'title' in doc and doc['title']:
            text_content.append(doc['title'])
        if 'abstract' in doc and doc['abstract']:
            text_content.append(doc['abstract'])
        
        prepared_doc = {
            'docno': doc['docno'],
            'text': ' '.join(text_content).strip()
        }
        yield prepared_doc

if not os.path.exists(index_loc):   
    print(f"Creating index at {index_loc}...")
    indexer = pt.IterDictIndexer(index_loc, meta=['docno', 'text'])
    indexer.index(prepare_docs(dataset.get_corpus_iter()))
    print(f"Index created at {index_loc}")

indexref = pt.IndexFactory.of(index_loc)
bm25 = pt.terrier.Retriever(indexref, wmodel="BM25", metadata=['docno', 'text'])

#%% Colbert implementation
colbert_factory = ColBERTFactory(
    colbert_model="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip",
    index_root=index_loc,
    index_name="trec_covid",
    gpu=True,  # change this if, like me, you don't have an Nvidia GPU :)
    )

colbert_pipeline = (bm25 % 100) >> doc_fetcher >> colbert_factory.text_scorer()

#%% SBERT implementation

sbert_reranker = SBERTReranker(alpha=0.2)

sbert_pipeline = (bm25 % 100) >> doc_fetcher >> sbert_reranker

# %% T5 implementation

monoT5 = MonoT5ReRanker(text_field='text')
# duoT5 = DuoT5ReRanker(text_field='abstract')

# Re-rank top 100 using monoT5.
monoT5_pipeline = (bm25 % 100) >> doc_fetcher >> monoT5 


#%% TasB implementation

tasB_pipeline = (bm25 % 100) >> doc_fetcher >> TasB.dot().text_scorer()

#%% -------------------------------------- TESTS ---------------------------------------------------
print("Running experiments...")
def run_experiment(query_type):
    pipelines = [bm25 % 100, colbert_pipeline % 100, sbert_pipeline % 100, monoT5_pipeline % 100, tasB_pipeline % 100]
    pipeline_names = ["BM25", "BM25 >> TCT-ColBERT % 100", "BM25 >> SBERT % 100", "BM25 >> MonoT5 % 100", "BM25 >> TasB % 100"]
    metrics = [RR @ 10, nDCG @ 10, MAP @ 100, 'mrt', R @ 10, R @ 25, R @ 50]

    results = {}

    for i, (pipeline, name) in enumerate(zip(pipelines, pipeline_names)):
        print(f"Running {name} on {query_type} queries...")
        result = pt.Experiment(
            [pipeline],
            dataset.get_topics(query_type),
            dataset.get_qrels(),
            eval_metrics=metrics,
            names=[name],
        )
        
        results[name] = {}
        for metric in metrics:
            metric_name = str(metric) if not isinstance(metric, str) else metric
            results[name][metric_name] = result.loc[0, metric_name]

    output_file = f"trec_covid_results_{query_type}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_file}")
    
    return results

print("Running title queries...")

if os.path.exists("trec_covid_results_title.json"):
    print("Loading title results from file...")
    title_results = json.load(open("trec_covid_results_title.json"))
else:
    title_results = run_experiment('title')

print("Running description queries...")
if os.path.exists("trec_covid_results_description.json"):
    print("Loading description results from file...")
    description_results = json.load(open("trec_covid_results_description.json"))
else:
    description_results = run_experiment('description')

print("Running narrative queries...")
if os.path.exists("trec_covid_results_narrative.json"):
    print("Loading narrative results from file...")
    narrative_results = json.load(open("trec_covid_results_narrative.json"))
else:
    narrative_results = run_experiment('narrative')

print("Creating visualizations...")

def plot_metrics(title_results, description_results, narrative_results=None):
    import numpy as np
    
    pipelines = ["BM25", "BM25 >> TCT-ColBERT % 100", "BM25 >> SBERT % 100", "BM25 >> MonoT5 % 100", "BM25 >> TasB % 100"]
    
    pipelines = [p for p in pipelines if p in title_results]
    
    metrics = list(title_results[pipelines[0]].keys())
    
    bar_width = 0.25
    r1 = np.arange(len(pipelines))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        title_values = [title_results[pipeline][metric] for pipeline in pipelines]
        desc_values = [description_results[pipeline][metric] for pipeline in pipelines]
        narr_values = [narrative_results[pipeline][metric] for pipeline in pipelines] if narrative_results else None
        
        plt.bar(r1, title_values, width=bar_width, label='Title', color='skyblue')
        plt.bar(r2, desc_values, width=bar_width, label='Description', color='lightgreen')
        if narr_values:
            plt.bar(r3, narr_values, width=bar_width, label='Narrative', color='salmon')
        
        plt.xlabel('Pipeline', fontweight='bold')
        plt.ylabel(metric, fontweight='bold')
        plt.title(f'Comparison of {metric} across Different Pipelines and Query Types')
        
        if narr_values:
            plt.xticks([r + bar_width for r in range(len(pipelines))], pipelines, rotation=45, ha='right')
        else:
            plt.xticks([r + bar_width/2 for r in range(len(pipelines))], pipelines, rotation=45, ha='right')
        
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f"trec_covid_{metric.replace('@', '_at_').replace('>', '_greater_')}_comparison.png")
        plt.close()
        
    print("Visualizations saved as PNG files.")

plot_metrics(title_results, description_results, narrative_results)

plt.figure(figsize=(10, 6))
metric = 'nDCG@10'

pipelines = ["BM25", "BM25 >> TCT-ColBERT % 100", "BM25 >> SBERT % 100", "BM25 >> MonoT5 % 100"]
title_values = [title_results[pipeline][metric] for pipeline in pipelines]
desc_values = [description_results[pipeline][metric] for pipeline in pipelines]
narr_values = [narrative_results[pipeline][metric] for pipeline in pipelines] if narrative_results else None

x = np.arange(len(pipelines))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width, title_values, width, label='Title', color='skyblue')
rects2 = ax.bar(x, desc_values, width, label='Description', color='lightgreen')
if narr_values:
    rects3 = ax.bar(x + width, narr_values, width, label='Narrative', color='salmon')

ax.set_ylabel(metric, fontsize=12, fontweight='bold')
ax.set_title(f'Comparison of {metric} across Different Pipelines and Query Types', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(pipelines, rotation=45, ha='right', fontsize=10, fontweight='bold')
ax.legend(fontsize=12)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

autolabel(rects1)
autolabel(rects2)
if narr_values:
    autolabel(rects3)

fig.tight_layout()
plt.savefig(f"trec_covid_{metric.replace('@', '_at_')}_highlight.png")
plt.close()


