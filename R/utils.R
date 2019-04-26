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
    similarities <- as.data.frame.table(similarities, stringsAsFactors = FALSE)
    colnames(similarities) <- c("term1", "term2", "similarity")
    similarities <- similarities[order(factor(similarities$term1), similarities$similarity, decreasing = TRUE), ]
    similarities$rank <- stats::ave(similarities$similarity, similarities$term1, FUN = seq_along)
    similarities <- similarities[similarities$rank <= top_n, ]
    rownames(similarities) <- NULL
  }
  similarities
}


#' @title Get the scale of embedding similarities alongside a Starspace model
#' @description Calculates embedding similarities between 2 embedding matrices and gets the range of resulting similarities.
#' @param x an object of class \code{textspace} as returned by \code{\link{starspace}} or \code{\link{starspace_load_model}}
#' @param from an embedding matrix. Defaults to the embeddings of all the labels and the words from the model.
#' @param to an embedding matrix. Defaults to the embeddings of all the labels.
#' @param probs numeric vector of probabilities ranging from 0-1. Passed on to \code{\link[stats]{quantile}}
#' @param breaks passed on to \code{\link[graphics]{hist}}
#' @param ... other parameters passed on to \code{\link[graphics]{hist}}
#' @return a list with elements 
#' \itemize{
#' \item{range: the range of the embedding similarities between \code{from} and \code{to}}
#' \item{quantile: the quantiles of the embedding similarities between \code{from} and \code{to}}
#' \item{hist: the histogram of the embedding similarities between \code{from} and \code{to}}
#' }
#' @export
#' @examples 
#' data(dekamer, package = "ruimtehol")
#' dekamer <- subset(dekamer, depotdat < as.Date("2017-02-01"))
#' dekamer$text <- strsplit(dekamer$question, "\\W")
#' dekamer$text <- lapply(dekamer$text, FUN = function(x) setdiff(x, ""))
#' dekamer$text <- sapply(dekamer$text, 
#'                        FUN = function(x) paste(x, collapse = " "))
#' dekamer$question_theme_main <- gsub(" ", "-", dekamer$question_theme_main)
#' 
#' set.seed(123456789)
#' model <- embed_tagspace(x = tolower(dekamer$text), 
#'                         y = dekamer$question_theme_main, 
#'                         early_stopping = 0.8, 
#'                         dim = 10, minCount = 5)
#' ranges <- range(model)
#' ranges$range
#' ranges$quantile
#' plot(ranges$hist, main = "Histogram of embedding similarities")                         
range.textspace <- function(x, from = as.matrix(x), to = as.matrix(x, type = "labels"), 
                            probs = seq(0, 1, by = 0.01), breaks = "scott", ...){
  similarities <- embedding_similarity(from, to, type = x$args$param$similarity)
  result <- list(range = range(similarities, na.rm = TRUE),
                 quantile = stats::quantile(similarities, probs = probs, na.rm = TRUE),
                 hist = graphics::hist(similarities, breaks = breaks, plot = FALSE))
  result
}


