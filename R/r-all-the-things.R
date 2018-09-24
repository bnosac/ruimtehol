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
#' model <- embed_words(x, dim = 15, ws = 7, epoch = 5, minCount = 5, ngrams = 1)
#' wordvectors <- as.matrix(model)
#' 
#' mostsimilar <- embedding_similarity(wordvectors, wordvectors["weekend", ])
#' head(sort(mostsimilar[, 1], decreasing = TRUE), 10)
#' mostsimilar <- embedding_similarity(wordvectors, wordvectors["vriendelijk", ])
#' head(sort(mostsimilar[, 1], decreasing = TRUE), 10)
#' mostsimilar <- embedding_similarity(wordvectors, wordvectors["grote", ])
#' head(sort(mostsimilar[, 1], decreasing = TRUE), 10)
embed_words <- function(x, model = "embed_words.bin", ...) {
  filename <- tempfile(pattern = "textspace_", fileext = ".txt")
  writeLines(text = paste(x, collapse = "\n"), con = filename)
  starspace(model = model, file = filename, trainMode = 5, ...)
}

#' @title NotYetImplemented
#' @description NotYetImplemented
#' @export
#' @return NotYetImplemented
embed_sentences <- function() {
  .NotYetImplemented()
  ## Each article contains several sentences with several words
  ## trainmode 3, fileFormat labelDoc
}

#' @title NotYetImplemented
#' @description NotYetImplemented
#' @export
#' @return NotYetImplemented
embed_articlespace <- function() {
  .NotYetImplemented()
  ## Each article contains several sentences
  ## trainmode 2, fileFormat labelDoc
  
  ## use case: if we have a new sentence, get me the article which looks like the sentence
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


if(FALSE){
  embed_entityrelations <- function() {
    .NotYetImplemented()
  }
  embed_images <- function() {
    .NotYetImplemented()
  }
  TagSpace <- embed_words
  SentenceSpace <- embed_sentences
  ArticleSpace <- embed_articles
  DocSpace <- embed_webpage
  PageSpace <- embed_clicks
  GraphSpace <- embed_entityrelations
  ImageSpace <- embed_images
}
