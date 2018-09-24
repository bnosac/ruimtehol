#' @title Cosine and Inner product based similarity 
#' @description Cosine and Inner product based similarity 
#' @param x a matrix with embeddings
#' @param y a matrix with embeddings
#' @param type either 'cosine' or 'dot'. If 'dot', returns inner-product based similarity, if 'cosine', returns cosine similarity
#' @export
#' @return similarity matrix between the rows of \code{x} and the rows of \code{y}. 
#' The similarity between row i of \code{x} and row j of \code{y} is found in cell \code{[i, j]} of the returned similarity matrix
#' @examples 
#' x <- matrix(rnorm(6), nrow = 2, ncol = 3)
#' rownames(x) <- c("word1", "word2")
#' y <- matrix(rnorm(15), nrow = 5, ncol = 3)
#' rownames(y) <- c("term1", "term2", "term3", "term4", "term5")
#' 
#' embedding_similarity(x, y, type = "cosine")
#' embedding_similarity(x, y, type = "dot")
embedding_similarity <- function(x, y, type = c("cosine", "dot")) {
  if(!is.matrix(x)){
    x <- matrix(x, nrow = 1)
  }
  if(!is.matrix(y)){
    y <- matrix(y, nrow = 1)
  }
  type <- match.arg(type)
  
  if(type == "dot"){
    tcrossprod(x, y)
  }else if(type == "cosine"){
    normx <- sqrt(rowSums(x^2))
    normy <- sqrt(rowSums(y^2))
    tcrossprod(x, y) / (normx %o% normy)
  }
}
