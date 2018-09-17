
#' @title Interface to Starspace
#' @description Interface to Starspace, providing raw access to the textspace functionality. For expert use only.
#' @param file the path to where the model file will be save. Defaults to 'textspace.bin'
#' @param ... UNDER CONSTRUCTION. Currently just look to the C++ code of the textspace function in the src/rcpp_textspace.cpp 
#' or type ruimtehol:::textspace_help() for seeing the Starspace options
#' @export
#' @return an object of class textspace
starspace <- function(file = "textspace.bin", ...) {
  file <- path.expand(file)
  object <- textspace(file = file, ...)
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

#' @title Load a Starspace model
#' @description Load a Starspace model
#' @param object either the path to a Starspace model on disk or an object of class \code{textspace} which you want to reload.
#' @export
#' @return an object of class textspace
starspace_load_model <- function(object){
  if(inherits(object, "textspace")){
    filename <- object$args$file
  }else{
    stopifnot(is.character(object))
    stopifnot(file.exists(object))
    filename <- object
  }
  object <- textspace_load_model(filename)
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


#' @title Get the document embeddings
#' @description Get the document embeddings
#' @param object an object of class \code{textspace} as returned by \code{\link{starspace}} or \code{\link{starspace_load_model}}
#' @param x character string with text
#' @param type the type of embedding requested. Currently only 'document' is possible
#' @export
#' @return a matrix 
starspace_embedding <- function(object, x, type = "document"){
  stopifnot(inherits(object, "textspace"))
  stopifnot(is.character(x))
  stopifnot(length(x) == 1)
  type <- match.arg(type)
  if(type == "document"){
    textspace_embedding_doc(object$model, x)  
  }else{
    .NotYetImplemented()
  }
}


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
