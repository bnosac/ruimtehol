## CHANGES IN ruimtehol VERSION 0.3.2

- Docs of starspace and starspace_dictionary: fix use of {} in itemize items
- Remove compliance.cpp and compliance.h and move the abort/exit statements directly in the cpp files
- Drop C++11 specification in Makevars 
  - hereby replacing use of std::random_shuffle with the Fisher-Yates Shuffle Algorithm as indicated at https://gallery.rcpp.org/articles/stl-random-shuffle/ - as data is reshuffled, model training will give different results compared to previous versions due to different randomisations of the training data

## CHANGES IN ruimtehol VERSION 0.3.1

- Changes in src/Starspace/src/model.cpp (EmbedModel::train). On Windows, no longer use threads as on CRAN that seems to make the package FAIL.

## CHANGES IN ruimtehol VERSION 0.3

- Fixed a bug in saving models trained with ngrams > 1. Embeddings of hashed buckets were not saved. 
- Default of bucket argument to the textspace C++ function is now changed to 100000 instead of 2000000, impacting all models with ngrams > 1. This was done as the embeddings of the buckets are now saved 
- as.matrix.textspace now also allows to get the LHS and RHS embeddings

## CHANGES IN ruimtehol VERSION 0.2.5

- Make example conditionally on availability of udpipe 

## CHANGES IN ruimtehol VERSION 0.2.4

- Changes to vignette: specify maxTrainTime in examples to avoid issues at CRAN caused by randomisation of starting values.

## CHANGES IN ruimtehol VERSION 0.2.3

- Changes to src/Makevars
    - Changes in src/Starspace/src/model.cpp (EmbedModel::train). On Mac, no longer use threads as on CRAN that seems to make the package FAIL (while on Travis it did not - see issue #10)

## CHANGES IN ruimtehol VERSION 0.2.2

- fix for embed_sentencespace & embed_docspace & embed_articlespace, sentences should be separated by "\t", not " \t " in order to avoid having "" in the dictionary

## CHANGES IN ruimtehol VERSION 0.2.1

- remove the GNU make as part of the SystemRequirements

## CHANGES IN ruimtehol VERSION 0.2

- Allow to do transfer learning by passing an embedding matrix and keep on training based on that matrix 
- Allow to do semi-supervised learning easily with embed_tagspace
- Attributes attached to a model are now also restored when loading a model with starspace_load_model of type 'ruimtehol'
- Add range.textspace to get the range of embedding similarities
- starspace now also returns the loss before training instead of only starting from epoch 1

## CHANGES IN ruimtehol VERSION 0.1.2

- Changes to src/Makevars
    - Added -pthread in PKG_CPPFLAGS and removed usage of SHLIB_PTHREAD_FLAGS

## CHANGES IN ruimtehol VERSION 0.1.1

- Initial release based on STARSPACE-2017-2.
