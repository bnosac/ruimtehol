
#' @title Interface to Starspace for training a Starspace model
#' @description Interface to Starspace for training a Starspace model, providing raw access to the C++ functionality. For expert use only.
#' @param model the full path to where the model file will be saved. Defaults to 'textspace.bin'.
#' @param file the full path to the file on disk which will be used for training.
#' @param trainMode integer with the training mode. Possible values are 0, 1, 2, 3, 4 or 5. Defaults to 0. The use cases are
#' \itemize{
#' \item 0: tagspace (classification tasks) and search tasks
#' \item 1: pagespace & docspace (content-based or collaborative filtering-based recommendation)
#' \item 2: articlespace (sentences within document)
#' \item 3: sentence embeddings and entity similarity 
#' \item 4: multi-relational graphs
#' \item 5: word embeddings 
#' }
#' @param fileFormat either one of 'fastText' or 'labelDoc'. See the documentation of StarSpace
#' @param ... arguments passed on to ruimtehol:::textspace. See the details below.
#' @references \url{https://github.com/facebookresearch}
#' @details
#' The internal function ruimtehol:::textspace allows direct access to the C++ code in order to run Starspace. 
#' The following arguments are available in that functionality when you do the training. Default settings are shown next to the definition: \cr
#' 
#' \strong{Arguments which define how the training is done:}
#' \itemize{
#' \item lr:              learning rate [0.01]
#' \item dim:             size of embedding vectors [100]
#' \item epoch:           number of epochs [5]
#' \item maxTrainTime:    max train time (secs) [8640000]
#' \item negSearchLimit:  number of negatives sampled [50]
#' \item maxNegSamples:   max number of negatives in a batch update [10]
#' \item loss:            loss function {hinge, softmax} [hinge]
#' \item margin:          margin parameter in hinge loss. It's only effective if hinge loss is used. [0.05]
#' \item similarity:      takes value in [cosine, dot]. Whether to use cosine or dot product as similarity function in  hinge loss. It's only effective if hinge loss is used. [cosine]
#' \item adagrad:         whether to use adagrad in training [1]
#' \item shareEmb:        whether to use the same embedding matrix for LHS and RHS. [1]
#' \item ws:              only used in trainMode 5, the size of the context window for word level training. [5]
#' \item dropoutLHS:      dropout probability for LHS features. [0]
#' \item dropoutRHS:      dropout probability for RHS features. [0]
#' \item initRandSd:      initial values of embeddings are randomly generated from normal distribution with mean=0, standard deviation=initRandSd. [0.001]
#' \item trainWord:       whether to train word level together with other tasks (for multi-tasking). [0]
#' \item wordWeight:      if trainWord is true, wordWeight specifies example weight for word level training examples. [0.5]
#' }
#' 
#' \strong{Arguments specific to the dictionary of words and labels:}
#' \itemize{
#' \item minCount:        minimal number of word occurences [1]
#' \item minCountLabel:   minimal number of label occurences [1]
#' \item ngrams:          max length of word ngram [1]
#' \item bucket:          number of buckets [2000000]
#' \item label:           labels prefix [__label__]
#' }
#' 
#' \strong{Arguments which define early stopping or proceeding of model building:}
#' \itemize{
#' \item initModel:       if not empty, it loads a previously trained model in -initModel and carry on training.
#' \item validationFile:  validation file path
#' \item validationPatience:    number of iterations of validation where does not improve before we stop training [10]
#' \item saveEveryEpoch:  save intermediate models after each epoch [0]
#' \item saveTempModel:   save intermediate models after each epoch with an unique name including epoch number [0]
#' }
#' @export
#' @return an object of class textspace
starspace <- function(model = "textspace.bin", file, trainMode = 0, fileFormat = c("fastText", "labelDoc"), ...) {
  file <- path.expand(file)
  stopifnot(trainMode %in% 0:5 && length(trainMode) == 1)
  fileFormat <- match.arg(fileFormat)
  ldots <- list(...)
  wrong <- intersect(c("testFile", "basedoc", "predictionFile", "K", "excludeLHS"), names(ldots))
  if(length(wrong)){
    stop(sprintf("You should not pass the arguments %s as they can only be used when doing starspace_test", paste(wrong, collapse = ", ")))
  }
  object <- textspace(model = model, trainFile = file, trainMode = as.integer(trainMode), fileFormat = fileFormat, ...)
  class(object) <- "textspace"
  object
}

#' @export
print.textspace <- function(x, ...){
  cat("Object of class textspace", sep = "\n")
  fsize <- file.info(x$args$file)$size
  cat(sprintf(" model saved at %s", x$args$file) , sep = "\n")
  cat(sprintf(" size of the model in Mb: %s", round(fsize / (2^20), 2)), sep = "\n")
  cat(sprintf(" dimension of the embedding: %s", x$args$dim) , sep = "\n")
  params <- mapply(key = names(x$args$param), value = x$args$param, FUN=function(key, value){
    sprintf("%s: %s", key, value)
  }, SIMPLIFY = TRUE)
  cat(sprintf(" training arguments:\n      %s", paste(params, collapse = "\n      ")), sep = "\n")
}

#' @title Get the dictionary of a Starspace model
#' @description Get the dictionary of a Starspace model
#' @param object an object of class \code{textspace} as returned by \code{\link{starspace}} or \code{\link{starspace_load_model}}
#' @export
#' @return a list with elements 
#' \enumerate{
#' \item{ntokens: }{The number of tokens in the data}
#' \item{nwords: }{The number of words which are part of the dictionary}
#' \item{nlabels: }{The number of labels which are part of the dictionary}
#' \item{labels: }{A character vector with the labels}
#' \item{dictionary: }{A data.frame with all the words and labels from the dictionary. This data.frame has columns term, is_word and is_label indicating
#' for each term if it is a word or a label}
#' }
starspace_dictionary <- function(object){
  stopifnot(inherits(object, "textspace"))
  textspace_dictionary(object$model)
}


#' @title Predict using a Starspace model 
#' @description Predict using a Starspace model 
#' @param object an object of class \code{textspace} as returned by \code{\link{starspace}} or \code{\link{starspace_load_model}}
#' @param newdata a character string of length 1
#' @param sep character string used to split \code{newdata} using boost::split
#' @param basedoc optional, the path to a file in labelDoc format, containing basedocs which are set of possible things to predict, if different than 
#' the ones from the training data
#' @param ... not used
#' @export
#' @return a list with elements input and a data.frame called prediction which has columns called label and similarity
predict.textspace <- function(object, newdata, sep = " ", basedoc, ...){
  stopifnot(is.character(newdata))
  stopifnot(length(newdata) == 1)
  stopifnot(nchar(newdata) > 0)
  if(object$args$data$trainMode != 0){
    warning("Using predict on model which was trained with another trainMode than 0.")
  }
  if(missing(basedoc)){
    textspace_predict(object$model, input = newdata, sep = sep)  
  }else{
    stopifnot(file.exists(basedoc))
    textspace_predict(object$model, input = newdata, sep = sep, basedoc = basedoc)  
  }
}


#' @title K-nearest neighbours using a Starspace model 
#' @description K-nearest neighbours using a Starspace model 
#' @param object an object of class \code{textspace} as returned by \code{\link{starspace}} or \code{\link{starspace_load_model}}
#' @param newdata a character string of length 1
#' @param k integer with the number of nearest neighbours
#' @param ... not used
#' @export
#' @return a list with elements input and a data.frame called prediction which has columns called label and prob
starspace_knn <- function(object, newdata, k = 5, ...){
  stopifnot(inherits(object, "textspace"))
  stopifnot(is.character(newdata))
  stopifnot(length(newdata) == 1)
  stopifnot(nchar(newdata) > 0)
  k <- as.integer(k)
  textspace_knn(object$model, newdata, k)
}

#' @title Load a Starspace model
#' @description Load a Starspace model
#' @param object either the path to a Starspace model on disk or an object of class \code{textspace} which you want to reload.
#' @param is_tsv logical indicating that if \code{object} is a file on disk, it is a tab-separated flat file. 
#' Defaults to \code{FALSE} indicating it is binary file as created by a call to \code{\link{starspace}}
#' @export
#' @return an object of class textspace
starspace_load_model <- function(object, is_tsv = FALSE){
  if(inherits(object, "textspace")){
    filename <- object$args$file
    is_tsv <- FALSE
  }else{
    stopifnot(is.character(object))
    stopifnot(file.exists(object))
    filename <- object
  }
  object <- textspace_load_model(filename, is_tsv)
  class(object) <- "textspace"
  object
}

#' @title Save a starspace model as a tab-delimited TSV file
#' @description Save a starspace model as a tab-delimited TSV file
#' @param object an object of class \code{textspace} as returned by \code{\link{starspace}} or \code{\link{starspace_load_model}}
#' @param file character string with the path to the file where to save the model
#' @export
#' @return invisibly, the \code{file} with the location of the TSV file
starspace_save_model <- function(object, file = "textspace.tsv"){
  stopifnot(inherits(object, "textspace"))
  textspace_save_model(object$model, file)
  invisible(file)
}


#' @title Get the document or ngram embeddings
#' @description Get the document or ngram embeddings
#' @param object an object of class \code{textspace} as returned by \code{\link{starspace}} or \code{\link{starspace_load_model}}
#' @param x character vector with text to get the embeddings 
#' \itemize{
#' \item If \code{type} is set to 'document', will assume that a space followed by a tab is used as separator of the sentences of each element of \code{x}.
#' \item If \code{type} is set to 'ngram', will assume that a space is used as separator of the words in case \code{x} contains words.
#' }
#' @param type the type of embedding requested. Either one of 'document' or 'ngram'. In case of document, 
#' the function returns the document embedding, in case of ngram the function returns the embedding of the 
#' provided ngram term which is used in the model building. 
#' @export
#' @return a matrix of embeddings
starspace_embedding <- function(object, x, type = c("document", "ngram")){
  stopifnot(inherits(object, "textspace"))
  stopifnot(is.character(x))
  type <- match.arg(type)
  if(type == "document"){
    textspace_embedding_doc(object$model, x)  
  }else if(type == "ngram"){
    textspace_embedding_ngram(object$model, x)  
  }
}


#' @export
as.matrix.textspace <- function(x, ...){
  embedding_dimension <- x$args$dim
  filename <- tempfile()
  starspace_save_model(x, file = filename)
  x <- utils::read.delim(filename, header = FALSE, stringsAsFactors = FALSE, encoding = "UTF-8", colClasses = c("character", rep("numeric", embedding_dimension)))
  dn <- list(x$V1, 1:(ncol(x)-1))
  x <- as.matrix(x[, -1, drop = FALSE])
  dimnames(x) <- dn
  x
}
