# ruimtehol: R package to Embed All the Things! using StarSpace

This repository contains an R package which wraps the StarSpace C++ library (https://github.com/facebookresearch/StarSpace), allowing the following:

- Text classification
- Learning word, sentence or document level embeddings
- Finding sentence or document similarity
- Ranking web documents
- Content-based or Collaborative filtering-based Recommendation, e.g. recommending music or videos.

<img src="vignettes/logo-ruimtehol.png" width="600">



## Installation

This package is not on CRAN, you can only install it as follows: `devtools::install_github("bnosac/ruimtehol", build_vignettes = TRUE)`


## Main functionalities

This R package allows to *Build Starspace models* on your own text / *Get embeddings* of words/ngrams/sentences/documents / Get *predictions* from a model (e.g. classification / ranking) / Get *nearest neighbours similarity*

The following functions are made available.

| Function                      | Functionality                                                  |
|-------------------------------|----------------------------------------------------------------|
| `starspace`                   | Low-level interface to build a Starspace model                 |
| `starspace_load_model`        | Load a pre-trained model or a tab-separated file               |
| `starspace_save_model`        | Save a Starspace model to a tab-separated file                 |
| `starspace_embedding`         | Get embeddings of documents/words/ngrams/labels                |
| `starspace_knn`               | Find k-nearest neighbouring information for new text           |
| `starspace_dictonary`         | Get words/labels with the model dictionary                     |
| `predict.textspace`           | Get predictions along a Starspace model                        |
| `as.matrix`                   | Get words and labels embeddings                                |
| `embedding_similarity`        | Basic cosine/dot product based similarity of embeddings        |
| `embed_wordspace`             | Build a Starspace model which calculates word/ngram embeddings                              |
| `embed_sentencespace`         | Build a Starspace model which calculates sentence embeddings                                |
| `embed_articlespace`          | Build a Starspace model for embedding an article and finding sentences-article similarities |
| `embed_tagspace`              | Build a Starspace model for multi-label classification                                      |
| `embed_docspace`              | Build a Starspace model for content-based recommendation                                    |
| `embed_pagespace`             | Build a Starspace model for interest-based recommendation                                   |
| `embed_entityrelationspace`   | Build a Starspace model for entity relationship completion (still under construction)       |



## Example


### Short example showing word embeddings


```r
library(ruimtehol)

## Get some training data
download.file("https://s3.amazonaws.com/fair-data/starspace/wikipedia_train250k.tgz", "wikipedia_train250k.tgz")
x <- readLines("wikipedia_train250k.tgz", encoding = "UTF-8")
x <- x[sample(x = length(x), size = 10000)]
writeLines(text = x, sep = "\n", con = "wikipedia_train10k.txt")
```

```r
## Train
model <- starspace(file = "wikipedia_train10k.txt", fileFormat = "labelDoc", dim = 10, trainMode = 3)
model

Object of class textspace
 model saved at textspace.bin
 size of the model in Mb: 0.87
 dimension of the embedding: 10
 training arguments:
      loss: hinge
      margin: 0.05
      similarity: cosine
      epoch: 5
      adagrad: TRUE
      lr: 0.01
      termLr: 1e-09
      norm: 1
      maxNegSamples: 10
      negSearchLimit: 50
      p: 0.5
      shareEmb: TRUE
      ws: 5
      dropoutLHS: 0
      dropoutRHS: 0
      initRandSd: 0.001
```

```r
embedding <- as.matrix(model)
embedding[c("school", "house"), ]

               1           2            3         4           5          6          7          8          9        10
school 0.0201249 -0.00478271 -0.018693000 0.0155070  0.01113670 -0.0184385 0.00892674 0.00549661 -0.0144082 0.0056668
house  0.0123371  0.01406140 -0.000166073 0.0313477 -0.00962703 -0.0237911 0.00225086 0.03393420  0.0035634 0.0160656
```

```r
## Save trained model as TSV so that you can inspect the embeddings e.g. with data.table::fread("wikipedia_embeddings.tsv")
starspace_save_model(model, file = "wikipedia_embeddings.tsv")

## Load a pre-trained model
model <- starspace_load_model("textspace.bin")

## Get the document embedding
starspace_embedding(model, "The apps to predict / get nearest neighbours are still under construction.")

           [,1]      [,2]        [,3]       [,4]       [,5]       [,6]       [,7]     [,8]      [,9]      [,10]
[1,] -0.4213823 0.4987145 -0.08066317 -0.6519815 -0.1743725 0.09401496 0.02670185 0.262726 0.1761705 0.04599866

starspace_knn(model, "What does this bunch of text look like", k = 10)
starspace_dictionary(model)
```

### Short example showing classification modelling (tagspace)


Below Starspace is used for classification

```r
library(fastrtext)
library(ruimtehol)
data(train_sentences, package = "fastrtext")

filename <- tempfile()
writeLines(text = paste(paste0("__label__", train_sentences$class.text),  tolower(train_sentences$text)),
           con = filename)

model <- starspace(file = filename, 
                   trainMode = 0, label = "__label__", 
                   similarity = "dot", verbose = TRUE, initRandSd = 0.01, adagrad = FALSE, 
                   ngrams = 1, lr = 0.01, epoch = 5, thread = 20, dim = 10, negSearchLimit = 5, maxNegSamples = 3)
predict(model, "We developed a two-level machine learning approach that in the first level considers two different 
                properties important for protein-protein binding derived from structural models of V3 and V3 sequences.")                   
```

## Notes

- Why did you call the package ruimtehol? Because that is the translation of StarSpace in WestVlaams.
- The R wrapper is distributed under the Mozilla Public License 2.0. The package contains a copy of the StarSpace C++ code (namely all code under src/Starspace) which has a BSD license (which is available in file LICENSE.notes) and also has an accompanying PATENTS file which you can inspect [here](inst/PATENTS).

## Support in text mining

Need support in text mining?
Contact BNOSAC: http://www.bnosac.be
