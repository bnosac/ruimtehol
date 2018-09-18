#' @title NotYetImplemented
#' @description NotYetImplemented
#' @export
#' @return NotYetImplemented
embed_words <- function() {
  .NotYetImplemented()
  ## trainmode does not even needs to be given, fileFormat neither
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
