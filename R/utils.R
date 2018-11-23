#' @title Cosine and Inner product based similarity 
#' @description Cosine and Inner product based similarity 
#' @param x a matrix with embeddings providing embeddings for words/n-grams/documents/labels as indicated in the rownames of the matrix
#' @param y a matrix with embeddings providing embeddings for words/n-grams/documents/labels as indicated in the rownames of the matrix
#' @param type either 'cosine' or 'dot'. If 'dot', returns inner-product based similarity, if 'cosine', returns cosine similarity
#' @param top_n integer indicating to return only the top n most similar terms from \code{y} for each row of \code{x}.
#' If \code{top_n} is supplied, a data.frame will be returned with only the highest similarities between \code{x} and \code{y} instead of all pairwise similarities
#' @export
#' @return 
#' By default, the function returns a similarity matrix between the rows of \code{x} and the rows of \code{y}. 
#' The similarity between row i of \code{x} and row j of \code{y} is found in cell \code{[i, j]} of the returned similarity matrix.\cr
#' If \code{top_n} is provided, the return value is a data.frame with columns term1, term2, similarity and rank 
#' indicating the similarity between the provided terms in \code{x} and \code{y} 
#' ordered from high to low similarity and keeping only the top_n most similar records.
#' @examples 
#' x <- matrix(rnorm(6), nrow = 2, ncol = 3)
#' rownames(x) <- c("word1", "word2")
#' y <- matrix(rnorm(15), nrow = 5, ncol = 3)
#' rownames(y) <- c("term1", "term2", "term3", "term4", "term5")
#' 
#' embedding_similarity(x, y, type = "cosine")
#' embedding_similarity(x, y, type = "dot")
#' embedding_similarity(x, y, type = "cosine", top_n = 1)
#' embedding_similarity(x, y, type = "dot", top_n = 1)
#' embedding_similarity(x, y, type = "cosine", top_n = 2)
#' embedding_similarity(x, y, type = "dot", top_n = 2)
#' embedding_similarity(x, y, type = "cosine", top_n = +Inf)
#' embedding_similarity(x, y, type = "dot", top_n = +Inf)
embedding_similarity <- function(x, y, type = c("cosine", "dot"), top_n = +Inf) {
  if(!is.matrix(x)){
    x <- matrix(x, nrow = 1)
  }
  if(!is.matrix(y)){
    y <- matrix(y, nrow = 1)
  }
  type <- match.arg(type)
  
  if(type == "dot"){
    similarities <- tcrossprod(x, y)
  }else if(type == "cosine"){
    normx <- sqrt(rowSums(x^2))
    normy <- sqrt(rowSums(y^2))
    similarities <- tcrossprod(x, y) / (normx %o% normy)
  }
  if(!missing(top_n)){
    similarities <- as.data.frame.table(similarities)
    colnames(similarities) <- c("term1", "term2", "similarity")
    similarities <- similarities[order(similarities$term1, similarities$similarity, decreasing = TRUE), ]
    similarities$rank <- stats::ave(similarities$similarity, similarities$term1, FUN = seq_along)
    similarities <- similarities[similarities$rank <= top_n, ]
    rownames(similarities) <- NULL
  }
  similarities
}



ruimtehol_save_model <- function(object, 
                                 labels = data.frame(code = character(), 
                                                     label = character(), stringsAsFactors = FALSE), 
                                 file = "textspace.ruimtehol"){
  stopifnot(inherits(object, "textspace"))
  stopifnot(inherits(labels, "data.frame"))
  stopifnot(all(c("code", "label") %in% colnames(labels)))
  ruimte <- list(object = object,
       labels = labels,
       embeddings = as.matrix(object))
  saveRDS(ruimte, file)
  invisible(file)
}

ruimtehol_load_model <- function(file){
  stopifnot(file.exists(file))
  ruimte <- readRDS(file)
  model <- ruimte$object
  model$args$data$testFile <- NULL
  arguments <- c(file = model$args$file, dim = model$args$dim, 
                 model$args$data, model$args$param, model$args$dictionary, model$args$options)
  arguments <- as.list(arguments)
  arguments$embeddings <- ruimte$embeddings
  object <- do.call(starspace, arguments)
  object$labels <- ruimte$labels
  object$iter <- model$iter
  object$labels$label_starspace <- as.character(sapply(object$labels$code, FUN=function(code) paste(model$args$dictionary$label, code, sep = "")))
  object
}