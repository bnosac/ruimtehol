#' @title Build a Starspace model to be used for classification purposes
#' @description Build a Starspace model to be used for classification purposes
#' @param x a character vector of text where tokens are separated by spaces
#' @param y a character vector of classes to predict or a list with the same length of \code{x} with several classes for each respective element of \code{x}
#' @param model name of the model which will be saved, passed on to \code{\link{starspace}}
#' @param ... further arguments passed on to \code{\link{starspace}}
#' @export
#' @return an object of class \code{textspace} as returned by \code{\link{starspace}}.
#' @examples 
#' library(udpipe)
#' library(tokenizers)
#' data(brussels_listings, package = "udpipe")
#' x <- tokenize_words(brussels_listings$name)
#' x <- sapply(x, FUN = function(x) paste(x, collapse = " "))
#' model <- textspace_classify(x = x, y = brussels_listings$room_type, 
#'                             dim = 10, minCount = 5)
#' predict(model, "room close to centre gare du midi")
#' starspace_embedding(model, "room close to centre gare du midi")
textspace_classify <- function(x, y, model = "tagspace.bin", ...) {
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
#' model <- embed_words(x, dim = 25, ws = 7, loss = "softmax", epoch = 20)
#' wordvectors <- as.matrix(model)
#' 
#' mostsimilar <- tcrossprod(wordvectors, wordvectors["weekend", , drop = FALSE])
#' head(sort(mostsimilar[, 1], decreasing = TRUE), 10)
#' mostsimilar <- tcrossprod(wordvectors, wordvectors["vriendelijk", , drop = FALSE])
#' head(sort(mostsimilar[, 1], decreasing = TRUE), 10)
#' mostsimilar <- tcrossprod(wordvectors, wordvectors["grote", , drop = FALSE])
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
embed_articles <- function() {
  .NotYetImplemented()
  ## Each article contains several sentences
  ## trainmode 2, fileFormat labelDoc
  
  ## use case: if we have a new sentence, get me the article which looks like the sentence
}

#' @title NotYetImplemented
#' @description NotYetImplemented
#' @export
#' @return NotYetImplemented
embed_webpage <- function() {
  .NotYetImplemented()
  ## user clicks on a web page which has content
  ## trainMode 1, fileFormat labelDoc
}

#' @title NotYetImplemented
#' @description NotYetImplemented
#' @export
#' @return NotYetImplemented
embed_clicks <- function() {
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
