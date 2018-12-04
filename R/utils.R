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
    similarities <- similarities[order(similarities$term1, similarities$similarity, decreasing = TRUE), ]
    similarities$rank <- stats::ave(similarities$similarity, similarities$term1, FUN = seq_along)
    similarities <- similarities[similarities$rank <= top_n, ]
    rownames(similarities) <- NULL
  }
  similarities
}


#' @title Add similarities of your text to the labels of a Starspace model
#' @description Add similarities of your text to the labels of a Starspace model. Similarities are computed with \code{\link{embedding_similarity}}
#' @param object an object of class \code{textspace} as returned by \code{\link{starspace}} or \code{\link{starspace_load_model}}
#' @param newdata a data frame with columns \code{doc_id} and \code{text} indicating the text for which you want to get the similarity to the labels of the Starspace model
#' @param type either 'cosine' or 'dot'. If 'dot', uses inner-product based similarity, if 'cosine', returns cosine similarity.
#' Passed on to \code{\link{embedding_similarity}}
#' @export
#' @return 
#' The data frame \code{newdata} where columns are added - one for each label indicating the similarity between the text and each of the labels.
#' @examples 
#' data(dekamer, package = "ruimtehol")
#' dekamer$text <- gsub("\\.([[:digit:]]+)\\.", ". \\1.", x = dekamer$question)
#' dekamer$text <- strsplit(dekamer$text, "\\W")
#' dekamer$text <- lapply(dekamer$text, FUN = function(x) setdiff(x, ""))
#' dekamer$text <- sapply(dekamer$text, 
#'                        FUN = function(x) paste(x, collapse = " "))
#' 
#' idx <- sample(nrow(dekamer), size = round(nrow(dekamer) * 0.9))
#' traindata <- dekamer[idx, ]
#' testdata <- dekamer[-idx, ]
#' model <- embed_tagspace(x = traindata$text, 
#'                         y = traindata$question_theme_main, 
#'                         early_stopping = 0.8,
#'                         dim = 10, minCount = 5)
#' scores <- cbind_embedding_similarity(model, testdata)
cbind_embedding_similarity <- function(object, newdata, type = c("cosine", "dot")) {
  type <- match.arg(type)
  stopifnot(inherits(object, "textspace"))
  stopifnot(is.data.frame(newdata))
  stopifnot(all(c("doc_id", "text") %in% colnames(newdata)))
  if(!requireNamespace("data.table", quietly = TRUE)){
    stop("embedding_similarity_labels requires the data.table package, which you can install from cran with install.packages('data.table')")
  }
  
  
  ## get dictionary
  d <- starspace_dictionary(object)
  if(length(d$labels) == 0){
    stop("You did not train the Starspace model with labels")
  }
  ## get embedding of the labels
  emb_labels <- starspace_embedding(object = object, x = d$labels, type = "ngram")
  rownames(emb_labels) <- remove_label_prefix(object, rownames(emb_labels))

  ## get similarities of the text with the text with the labels
  newdata      <- data.table::setDT(newdata)
  textvectors  <- starspace_embedding(object, x = unique(newdata$text), type = "document")
  similarities <- embedding_similarity(textvectors, emb_labels, top_n = d$nlabels, type = type)
  similarities <- data.table::setDT(similarities)
  similarities <- data.table::dcast.data.table(data = similarities, formula = term1 ~ term2, value.var = "similarity")
  similarities <- merge(newdata, similarities, by.x = "text", by.y = "term1", sort = FALSE)
  similarities <- data.table::setDF(similarities)
  similarities
}

remove_label_prefix <- function(object, x){
  length_label_prefix <- nchar(object$args$dictionary$label)
  ifelse(substr(x, 1, length_label_prefix) == object$args$dictionary$label,
         substr(x, length_label_prefix + 1L, nchar(x)),
         x)
}
