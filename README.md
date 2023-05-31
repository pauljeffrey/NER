## Introduction
This repository contains solutions to the problem challenge including the files for deploying model online.
It uses an NER system to retrieve specific information for any given drug from a database.

It takes a query text and using the contextual understanding of a bert model, it extraacts all the useful information from the text and uses itto 
query the database for results.

Do check out the jupyter notebook in this repo to understand how this system was built.

## Data Exploration
After observing the dataset, I noticed several problems with the dataset:
- drugs have inappropriate medical conditions e.g spironolactone is a diuretic and not used for acne
- There were inconsistencies in how the content of each feature sample were given (different formats)
- Some features are irrelevant for the task.
- some features contain null values.

## Data Processing
Most of the datafeatures contained  inappropriate values so, a script was written to fetch the right values for each.
For others, the mode value for that feature was used.


## Final Schema
The final data stored in parquet format consists of the following features:
- Drug_name : Str
- brand_name: Str
- medical_condition: Str
- Side_effect: Str
- generic_name: Str
- pregnancy_category: Str
- CSA: Str
- drug_link: Str


## Problem Statement
Possible cases with queries
- Just drug names are provided to the IR system e.g: amoxicillin
- full sentences/phrases can be provided (e.g "I want information on spironolactone", "Get doxycycline")
- Very complex queries could be provided.
- drug names might be mispelled : "parcetamol, doxycylin, amoxicilllin"
- drug abbreviations might be used : "PCM for paracetamol, diclo for diclofenac, IBP for ibuprofen "
- generic names or drug classes may be given instead of the drug names
- The drug specified might not be in the database.

Additionally, in this case, this retrieval system will definitely differ from others because most times clinicians or health workers want information for a particlar drug and not other similar drugs.

Therefore, we need a robust system to handle most of these problems. To do this, our system will need to recognize and extract medically named entities (e.g drug, side effect) and it has to be robust to spelling errors. 

We could use sophisticated large language models like BERT or ADA to run cosine or other similarity metrics on embeddings of the user's queries but that comes with a lot of problems intuitively. I mention some of them here:

- We might need to finetune this model as a lot of models have not been really trained on pharmaceutical data, however, our pharmaceutical data is small.  word embedding models are trained model but most do not really process pharmaceuticals word embedding is a trained model but most do not really process pharmaceuticals

- how does the model interprete wrongly spelt drugs? What does word embedding truly meanfor the model here? does it offer any semantics? word embedding models are trained using the words they are mostly associated with in a particular dataset. word embedding models are trained using the words they are mostly associated with in a particular dataset.

- What happens when their several drugs in a complex user query? e.g "I want information on paracetamol , orphenadrine alongside amlodipine. I need information on their side effects."

- What about drug abbreviations? how does it handle that? What if a drug is actually a phrase ? e.g paracetamol injection or efavirenz/ tenofovir?I mean word embedding is a trained model but most do not really process pharmaceuticals. What about drug abbreviations? how does it handle that? What if a drug is actually a phrase ? e.g paracetamol injection or efavirenz/ tenofovir?

In this case, that is difficult to model. Feeding " I want information on paracetamol" might give you a good embedding but how do we store drug embedding of the drug "paracetamol" without any context in our database? It might end up being a very expensive or inefficient process storing vector embeddings.
Is the word "paracetamol" just passed as input for conversion to an embedding or do we pass some context to it? In any of these cases, the retrieved results would be affected.


## Solution
To this effect, I focus instead on making an Named Entity Recognition + rule-based system to tackle this problems. With this model, we eliminate the use of vector dbs, hosting large models with large requirements online or paying for ada gpt subscriptions while just identifying drugs alongside some other identifiable drug attributes.

## Model Development
The core of the NER system is a finetuned bert based model called Distillbert-Cased that was finetuned on over 50k artificially generated dataset. It was trained on the task 
of Named Entity Recognition that involves predicting named entities for each tag.
Distillbert-Cased is a transformer based model.
Model could not be included here due to size (250mb)

## backend API implementation. 
Backend API makes use of FastAPI due to ease of implementation. It is a very simple implementation with one post function that receives user input in the following format:
- query : Str
- Focus : Str or List
- threshold: float (0-1) = 0.7 # Determines the level of confidence to control the model's predictions
- use_max: bool = None # whether to use the max prediction or the mode prediction

The system in API engine in turn returns the retrieved drugs alongside their information in a dictionary or json object.



# Inference Use
Model is yet to be deployed due to challenges faced during the deployment test phase. This is due to the dependency limitation during use of the transformers library.
