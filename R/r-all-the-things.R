#' @title Build a Starspace model to be used for classification purposes
#' @description Build a Starspace model to be used for classification purposes
#' @param x a character vector of text where tokens are separated by spaces
#' @param y a character vector of classes to predict or a list with the same length of \code{x} with several classes for each respective element of \code{x}
#' @param model name of the model which will be saved, passed on to \code{\link{starspace}}
#' @param ... further arguments passed on to \code{\link{starspace}}
#' @export
#' @return an object of class \code{textspace} as returned by \code{\link{starspace}}.
#' @examples 
#' library(tokenizers)
#' data(dekamer, package = "ruimtehol")
#' dekamer$text <- gsub("\\.([[:digit:]]+)\\.", ". \\1.", x = dekamer$question)
#' dekamer$text <- tokenize_words(dekamer$text)
#' dekamer$text <- sapply(dekamer$text, 
#'                        FUN = function(x) paste(x, collapse = " "))
#' 
#' model <- embed_tagspace(x = dekamer$text, 
#'                         y = dekamer$question_theme_main, 
#'                         dim = 10, minCount = 5)
#' predict(model, "de nmbs heeft het treinaanbod uitgebreid")
#' predict(model, "de migranten komen naar europa, in asielcentra ...")
#' starspace_embedding(model, "de nmbs heeft het treinaanbod uitgebreid")
#' starspace_embedding(model, "__label__MIGRATIEBELEID", type = "ngram")
embed_tagspace <- function(x, y, model = "tagspace.bin", ...) {
  ldots <- list(...)
  filename <- tempfile(pattern = "textspace_", fileext = ".txt")
  label <- "__label__"
  if("label" %in% names(ldots)){
    label <- ldots$label
  }
  if(is.list(y)){
    targets <- sapply(y, FUN=function(x) paste(paste(label, x, sep = ""), collapse = " "))
  }else{
    targets <- paste(label, y, sep = "")
  }
  writeLines(text = paste(targets, x), con = filename)
  on.exit(file.remove(filename))
  starspace(model = model, file = filename, trainMode = 0, label = label, fileFormat = "fastText", ...)
}

#' @title Build a Starspace model which calculates word embeddings
#' @description Build a Starspace model which calculates word embeddings
#' @param x a character vector of text where tokens are separated by spaces
#' @param model name of the model which will be saved, passed on to \code{\link{starspace}}
#' @param ... further arguments passed on to \code{\link{starspace}}
#' @export
#' @return an object of class \code{textspace} as returned by \code{\link{starspace}}.
#' @examples 
#' library(udpipe)
#' library(tokenizers)
#' data(brussels_reviews, package = "udpipe")
#' x <- subset(brussels_reviews, language == "nl")
#' x <- tokenize_words(x$feedback)
#' x <- sapply(x, FUN = function(x) paste(x, collapse = " "))
#' 
#' model <- embed_wordspace(x, dim = 15, ws = 7, epoch = 5, minCount = 5, ngrams = 1)
#' wordvectors <- as.matrix(model)
#' 
#' mostsimilar <- embedding_similarity(wordvectors, wordvectors["weekend", ])
#' head(sort(mostsimilar[, 1], decreasing = TRUE), 10)
#' mostsimilar <- embedding_similarity(wordvectors, wordvectors["vriendelijk", ])
#' head(sort(mostsimilar[, 1], decreasing = TRUE), 10)
#' mostsimilar <- embedding_similarity(wordvectors, wordvectors["grote", ])
#' head(sort(mostsimilar[, 1], decreasing = TRUE), 10)
embed_wordspace <- function(x, model = "embed_wordspace.bin", ...) {
  filename <- tempfile(pattern = "textspace_", fileext = ".txt")
  writeLines(text = paste(x, collapse = "\n"), con = filename)
  on.exit(file.remove(filename))
  starspace(model = model, file = filename, trainMode = 5, ...)
}

#' @title Build a Starspace model to be used for sentence embedding
#' @description Build a Starspace model to be used for sentence embedding
#' @param x a data.frame with sentences containg the columns doc_id, sentence_id and token 
#' The doc_id is just an article or document identifier, 
#' the sentence_id column is a character field which contains words which are separated by a space and should not contain any tab characters
#' @param model name of the model which will be saved, passed on to \code{\link{starspace}}
#' @param ... further arguments passed on to \code{\link{starspace}}
#' @export
#' @return an object of class \code{textspace} as returned by \code{\link{starspace}}.
#' @examples 
#' library(udpipe)
#' data(dekamer, package = "ruimtehol")
#' dekamer <- subset(dekamer, question_theme_main == "DEFENSIEBELEID")
#' x <- udpipe(dekamer$question, "dutch", tagger = "none", parser = "none", trace = 100)
#' x <- x[, c("doc_id", "sentence_id", "sentence", "token")]
#' model <- embed_sentencespace(x, dim = 15, epoch = 5, minCount = 5)
#' predict(model, "Wat zijn de cijfers qua doorstroming van 2016?", 
#'         basecode = unique(x$sentence))
#' 
#' embeddings <- starspace_embedding(model, unique(x$sentence), type = "document")
#' dim(embeddings)
#' 
#' sentence <- "Wat zijn de cijfers qua doorstroming van 2016?"
#' mostsimilar <- embedding_similarity(embeddings, embeddings[sentence, ])
#' head(sort(mostsimilar[, 1], decreasing = TRUE), 3)
embed_sentencespace <- function(x, model = "sentencespace.bin", ...) {
  stopifnot(is.data.frame(x))
  stopifnot(all(c("doc_id", "sentence_id", "token") %in% colnames(x)))
  filename <- tempfile(pattern = "textspace_", fileext = ".txt")
  x <- split(x, f = x$doc_id)
  x <- sapply(x, FUN=function(tokens){
    sentences <- split(tokens, tokens$sentence_id)
    sentences <- sapply(sentences, FUN=function(x) paste(x$token, collapse = " "))
    paste(sentences, collapse = " \t ")
  })
  writeLines(text = paste(x, collapse = "\n"), con = filename)
  on.exit(file.remove(filename))
  starspace(model = model, file = filename, trainMode = 3, fileFormat = "labelDoc", ...)
}

#' @title Build a Starspace model for learning the mapping between sentences and articles (articlespace)
#' @description Build a Starspace model for learning the mapping between sentences and articles (articlespace)
#' @param x a data.frame with sentences containg the columns doc_id, sentence_id and token 
#' The doc_id is just an article or document identifier, 
#' the sentence_id column is a character field which contains words which are separated by a space and should not contain any tab characters
#' @param model name of the model which will be saved, passed on to \code{\link{starspace}}
#' @param ... further arguments passed on to \code{\link{starspace}}
#' @export
#' @return an object of class \code{textspace} as returned by \code{\link{starspace}}.
#' @examples 
#' library(udpipe)
#' data(dekamer, package = "ruimtehol")
#' dekamer <- subset(dekamer, question_theme_main == "DEFENSIEBELEID")
#' x <- udpipe(dekamer$question, "dutch", tagger = "none", parser = "none", trace = 100)
#' x <- x[, c("doc_id", "sentence_id", "sentence", "token")]
#' model <- embed_articlespace(x, dim = 15, epoch = 5, minCount = 5)
#' 
#' embeddings <- starspace_embedding(model, unique(x$sentence), type = "document")
#' dim(embeddings)
#' 
#' sentence <- "Wat zijn de cijfers qua doorstroming van 2016?"
#' mostsimilar <- embedding_similarity(embeddings, embeddings[sentence, ])
#' head(sort(mostsimilar[, 1], decreasing = TRUE), 3)
embed_articlespace <- function(x, model = "articlespace.bin", ...) {
  stopifnot(is.data.frame(x))
  stopifnot(all(c("doc_id", "sentence_id", "token") %in% colnames(x)))
  filename <- tempfile(pattern = "textspace_", fileext = ".txt")
  x <- split(x, f = x$doc_id)
  x <- sapply(x, FUN=function(tokens){
    sentences <- split(tokens, tokens$sentence_id)
    sentences <- sapply(sentences, FUN=function(x) paste(x$token, collapse = " "))
    paste(sentences, collapse = " \t ")
  })
  writeLines(text = paste(x, collapse = "\n"), con = filename)
  on.exit(file.remove(filename))
  starspace(model = model, file = filename, trainMode = 2, fileFormat = "labelDoc", ...)
}

#' @title NotYetImplemented
#' @description NotYetImplemented
#' @export
#' @return NotYetImplemented
embed_docspace <- embed_webpage <- function() {
  .NotYetImplemented()
  ## user clicks on a web page which has content
  ## trainMode 1, fileFormat labelDoc
}

#' @title NotYetImplemented
#' @description NotYetImplemented
#' @export
#' @return NotYetImplemented
embed_pagespace <- embed_clicks <- function() {
  .NotYetImplemented()
  ## user clicks or is fan of a webpage
  ## trainMode 1
}


embed_entityrelationspace <- function() {
  .NotYetImplemented()
}
embed_imagespace <- function() {
  .NotYetImplemented()
}

if(FALSE){
  TagSpace <- embed_tagspace
  WordSpace <- embed_wordspace
  SentenceSpace <- embed_sentencespace
  ArticleSpace <- embed_articlespace
  DocSpace <- embed_webpage <- embed_docspace
  PageSpace <- embed_clicks <- embed_pagespace
  GraphSpace <- embed_entityrelationspace
  ImageSpace <- embed_imagespace
}
