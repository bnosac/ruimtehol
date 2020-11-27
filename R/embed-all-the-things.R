
#' @title Interface to Starspace for training a Starspace model
#' @description Interface to Starspace for training a Starspace model, providing raw access to the C++ functionality. 
#' @param model the full path to where the model file will be saved. Defaults to 'textspace.bin'.
#' @param file the full path to the file on disk which will be used for training.
#' @param trainMode integer with the training mode. Possible values are 0, 1, 2, 3, 4 or 5. Defaults to 0. The use cases are
#' \itemize{
#' \item 0: tagspace (classification tasks) and search tasks
#' \item 1: pagespace & docspace (interest-based or content-based recommendation)
#' \item 2: articlespace (sentences within document)
#' \item 3: sentence embeddings and entity similarity 
#' \item 4: multi-relational graphs
#' \item 5: word embeddings 
#' }
#' @param fileFormat either one of 'fastText' or 'labelDoc'. See the documentation of StarSpace
#' @param thread integer with the number of threads to use. Defaults to 1.
#' @param dim the size of the embedding vectors (integer, defaults to 100)
#' @param epoch number of epochs (integer, defaults to 5)
#' @param lr learning rate (numeric, defaults to 0.01)
#' @param loss loss function (either 'hinge' or 'softmax')
#' @param margin margin parameter in case of hinge loss (numeric, defaults to 0.05)
#' @param similarity cosine or dot product similarity in cas of hinge loss (character, defaults to 'cosine')
#' @param negSearchLimit number of negatives sampled (integer, defaults to 50)
#' @param adagrad whether to use adagrad in training (logical)
#' @param ws the size of the context window for word level training - only used in trainMode 5 (integer, defaults to 5)
#' @param minCount minimal number of word occurences for being part of the dictionary (integer, defaults to 1 keeping all words)
#' @param minCountLabel minimal number of label occurences for being part of the dictionary (integer, defaults to 1 keeping all labels)
#' @param ngrams max length of word ngram (integer, defaults to 1, using only unigrams)
#' @param label labels prefix (character string identifying how a label is prefixed, defaults to '__label__') 
#' @param ... arguments passed on to ruimtehol:::textspace. See the details below.
#' @references \url{https://github.com/facebookresearch}
#' @note  
#' The function \code{starspace} is a tiny wrapper over the internal function ruimtehol:::textspace which 
#' allows direct access to the C++ code in order to run Starspace. \cr
#' The following arguments are available in that functionality when you do the training. 
#' Default settings are shown next to the definition. Some of these arguments are directly set in the \code{starspace} function,
#' others can be passed on with ... . \cr
#' 
#' \strong{Arguments which define how the training is done:}
#' \itemize{
#' \item dim:             size of embedding vectors [100]
#' \item epoch:           number of epochs [5]
#' \item lr:              learning rate [0.01]
#' \item loss:            loss function {hinge, softmax} [hinge]
#' \item margin:          margin parameter in hinge loss. It's only effective if hinge loss is used. [0.05]
#' \item similarity:      takes value in [cosine, dot]. Whether to use cosine or dot product as similarity function in  hinge loss. It's only effective if hinge loss is used. [cosine]
#' \item negSearchLimit:  number of negatives sampled [50]
#' \item maxNegSamples:   max number of negatives in a batch update [10]
#' \item p:               normalization parameter: normalize sum of embeddings by dividing Size^p [0.5]
#' \item adagrad:         whether to use adagrad in training [1]
#' \item ws:              only used in trainMode 5, the size of the context window for word level training. [5]
#' \item dropoutLHS:      dropout probability for LHS features. [0]
#' \item dropoutRHS:      dropout probability for RHS features. [0]
#' \item shareEmb:        whether to use the same embedding matrix for LHS and RHS. [1]
#' \item initRandSd:      initial values of embeddings are randomly generated from normal distribution with mean=0, standard deviation=initRandSd. [0.001]
#' }
#' 
#' \strong{Arguments specific to the dictionary of words and labels:}
#' \itemize{
#' \item minCount:        minimal number of word occurences [1]
#' \item minCountLabel:   minimal number of label occurences [1]
#' \item ngrams:          max length of word ngram [1]
#' \item bucket:          number of buckets [100000]
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
#' \item maxTrainTime:    max train time (secs) [8640000]
#' }
#' 
#' \strong{Other:}
#' \itemize{
#' \item trainWord:       whether to train word level together with other tasks (for multi-tasking). [0]
#' \item wordWeight:      if trainWord is true, wordWeight specifies example weight for word level training examples. [0.5]
#' \item useWeight        whether input file contains weights [0]
#' }
#' @export
#' @return an object of class textspace which is a list with elements 
#' \itemize{
#' \item model: a Rcpp pointer to the model
#' \item args: a list with elements
#' \enumerate{
#' \item file: the binary file of the model saved on disk
#' \item dim: the dimension of the embedding
#' \item data: data-specific Starspace training parameters
#' \item param: algorithm-specific Starspace training parameters
#' \item dictionary: parameters which define ths dictionary of words and labels in Starspace
#' \item options: parameters specific to duration of training, the text preparation and the training batch size
#' \item test: parameters specific to model testing
#' }
#' \item iter: a list with element epoch, lr, error and error_validation showing the error after each epoch
#' }
#' @examples 
#' \dontrun{
#' data(dekamer, package = "ruimtehol")
#' x <- strsplit(dekamer$question, "\\W")
#' x <- lapply(x, FUN = function(x) x[x != ""])
#' x <- sapply(x, FUN = function(x) paste(x, collapse = " "))
#' 
#' idx <- sample.int(n = nrow(dekamer), size = round(nrow(dekamer) * 0.7))
#' writeLines(x[idx], con = "traindata.txt")
#' writeLines(x[-idx], con = "validationdata.txt")
#' 
#' set.seed(123456789)
#' m <- starspace(file = "traindata.txt", validationFile = "validationdata.txt", 
#'                trainMode = 5, dim = 10, 
#'                loss = "softmax", lr = 0.01, ngrams = 2, minCount = 5,
#'                similarity = "cosine", adagrad = TRUE, ws = 7, epoch = 3,
#'                maxTrainTime = 10)
#' str(starspace_dictionary(m))              
#' wordvectors <- as.matrix(m)
#' wv <- starspace_embedding(m, 
#'                           x = c("Nationale Loterij", "migranten", "pensioen"),
#'                           type = "ngram")
#' wv
#' mostsimilar <- embedding_similarity(wordvectors, wv["pensioen", ])
#' head(sort(mostsimilar[, 1], decreasing = TRUE), 10)
#' starspace_knn(m, "koning")
#' 
#' ## clean up for cran
#' file.remove(c("traindata.txt", "validationdata.txt"))
#' }
starspace <- function(model = "textspace.bin", file, trainMode = 0, fileFormat = c("fastText", "labelDoc"), label = "__label__", 
                      dim = 100,
                      epoch = 5,
                      lr = 0.01,
                      loss = c("hinge", "softmax"),
                      margin = 0.05,
                      similarity = c("cosine", "dot"),
                      negSearchLimit = 50,
                      adagrad = TRUE,
                      ws = 5,
                      minCount = 1,
                      minCountLabel = 1,
                      ngrams = 1,
                      thread = 1, ...) {
  ldots <- list(...)
  if(!"embeddings" %in% names(ldots)){
    file <- path.expand(file)  
  }else{
    if(missing(file)){
      file <- ""
    }
  }
  #else{
  #   if("embeddings_optimise" %in% names(ldots)){
  #     file <- path.expand(file) 
  #   }else{
  #     file <- ""  
  #   }
  # }
  stopifnot(trainMode %in% 0:5 && length(trainMode) == 1)
  fileFormat <- match.arg(fileFormat)
  loss <- match.arg(loss)
  similarity <- match.arg(similarity)
  #wrong <- intersect(c("testFile", "basedoc", "predictionFile", "K", "excludeLHS"), names(ldots))
  wrong <- intersect(c("testFile", "basedoc", "predictionFile", "excludeLHS"), names(ldots))
  if(length(wrong)){
    stop(sprintf("You should not pass the arguments %s as they can only be used when doing starspace_test", paste(wrong, collapse = ", ")))
  }
  ldots$model <- model
  ldots$trainFile <- file
  ldots$trainMode <- as.integer(trainMode)
  ldots$fileFormat <- fileFormat
  ldots$label <- label
  ldots$dim <- as.integer(dim)
  ldots$epoch <- as.integer(epoch)
  ldots$lr <- lr
  ldots$loss <- loss
  ldots$margin <- margin
  ldots$similarity <- similarity
  ldots$negSearchLimit <- as.integer(negSearchLimit)
  ldots$adagrad <- as.logical(adagrad)
  ldots$ws <- as.integer(ws)
  ldots$minCount <- as.integer(minCount)
  ldots$minCountLabel <- as.integer(minCountLabel)
  ldots$ngrams <- as.integer(ngrams)
  ldots$thread <- as.integer(thread)
  if(!"embeddings" %in% names(ldots)){
    ldots$embeddings <- matrix(data = numeric(), nrow = 0, ncol = as.integer(dim)) 
  }
  object <- do.call(textspace, ldots)
  #object <- textspace(model = model, trainFile = file, trainMode = as.integer(trainMode), 
  #                    fileFormat = fileFormat, label = label, dim = as.integer(dim), epoch = as.integer(epoch), 
  #                    lr = lr, loss = loss, margin = margin, similarity = similarity, 
  #                    negSearchLimit = as.integer(negSearchLimit), adagrad = as.logical(adagrad), 
  #                    ws = as.integer(ws), minCount = as.integer(minCount), 
  #                    minCountLabel = as.integer(minCountLabel), ngrams = as.integer(ngrams), 
  #                    thread = as.integer(thread), ...)
  class(object) <- "textspace"
  object
}

#' @export
print.textspace <- function(x, ...){
  cat("Object of class textspace", sep = "\n")
  if(file.exists(x$args$file)){
    fsize <- file.info(x$args$file)$size
    cat(sprintf(" model saved at %s", x$args$file) , sep = "\n")
    cat(sprintf(" size of the model in Mb: %s", round(fsize / (2^20), 2)), sep = "\n")  
  }
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
#' \item{dictionary_size: }{The size of the dictionary (nwords + nlabels)}
#' \item{dictionary: }{A data.frame with all the words and labels from the dictionary. This data.frame has columns term, is_word and is_label indicating
#' for each term if it is a word or a label}
#' }
#' @examples 
#' data(dekamer, package = "ruimtehol")
#' dekamer <- subset(dekamer, depotdat < as.Date("2017-02-01"))
#' dekamer$text <- strsplit(dekamer$question, "\\W")
#' dekamer$text <- lapply(dekamer$text, FUN = function(x) x[x != ""])
#' dekamer$text <- sapply(dekamer$text, 
#'                        FUN = function(x) paste(x, collapse = " "))
#' dekamer$question_theme_main <- gsub(" ", "-", dekamer$question_theme_main)
#' 
#' set.seed(123456789)
#' model <- embed_tagspace(x = tolower(dekamer$text), 
#'                         y = dekamer$question_theme_main, 
#'                         early_stopping = 0.8, 
#'                         dim = 10, minCount = 5)
#' dict <- starspace_dictionary(model)
#' str(dict)
starspace_dictionary <- function(object){
  stopifnot(inherits(object, "textspace"))
  textspace_dictionary(object$model)
}


#' @title Predict using a Starspace model 
#' @description The prediction functionality allows you to retrieve the following types of elements from a Starspace model:
#' \itemize{
#' \item \code{generic}: get general Starspace predictions in detail
#' \item \code{labels}: get similarity of your text to all the labels of the Starspace model
#' \item \code{embedding}: document embeddings of your text (shorthand for \code{\link{starspace_embedding}})
#' \item \code{knn}: k-nearest neighbouring (most similar) elements of the model dictionary compared to your input text (shorthand for \code{\link{starspace_knn}})
#' }
#' @param object an object of class \code{textspace} as returned by \code{\link{starspace}} or \code{\link{starspace_load_model}}
#' @param newdata a data frame with columns \code{doc_id} and \code{text} or a character vector with text where the names of the character vector represent an identifier of that text
#' @param type character string: either 'generic', 'labels', 'embedding', 'knn'. Defaults to 'generic'  
#' @param k integer with the number of predictions to make. Defaults to 5. Only used in case \code{type} is set to \code{'generic'} or \code{'knn'}
#' @param sep character string used to split \code{newdata} using boost::split. Only used in case \code{type} is set to \code{'generic'}
#' @param basedoc optional, either a character vector of possible elements to predict or 
#' the path to a file in labelDoc format, containing basedocs which are set of possible things to predict, if different than 
#' the ones from the training data. Only used in case \code{type} is set to \code{'generic'}
#' @param ... not used
#' @export
#' @return The following is returned, depending on the argument \code{type}:
#' \itemize{
#' \item In case type is set to \code{'generic'}: a list, one for each row or element in \code{newdata}. 
#' Each list element is a list with elements 
#' \itemize{
#' \item doc_id: the identifier of the text
#' \item text: the character string with the text
#' \item prediction: data.frame with columns label, label_starspace and similarity 
#' indicating the predicted label and the similarity of the text to the label
#' \item terms: a list with elements basedoc_index and basedoc_terms indicating the position in basedoc and the terms 
#' which are part of the dictionary which are used to find the similarity
#' }
#' \item In case type is set to \code{'labels'}: a data.frame is returned namely:\cr
#' The data.frame \code{newdata} where several columns are added, one for each label in the Starspace model. 
#' These columns contain the similarities of the text to the label. 
#' Similarities are computed with \code{\link{embedding_similarity}} indicating embedding similarities 
#' of the text compared to the labels using either cosine or dot product as was used during model training.
#' \item In case type is set to \code{'embedding'}: \cr
#' A matrix of document embeddings, one embedding for each text in \code{newdata} as returned by \code{\link{starspace_embedding}}. 
#' The rownames of this matrix are set to the document identifiers of \code{newdata}.
#' \item In case type is set to \code{'knn'}: a list of data.frames, one for each row or element in \code{newdata} \cr
#' Each of these data frames contains the columns doc_id, label, similarity and rank indicating the
#' k-nearest neighbouring (most similar) elements of the model dictionary compared to your input text as returned by \code{\link{starspace_knn}}
#' }
#' @examples
#' data(dekamer, package = "ruimtehol")
#' dekamer$text <- strsplit(dekamer$question, "\\W")
#' dekamer$text <- lapply(dekamer$text, FUN = function(x) x[x != ""])
#' dekamer$text <- sapply(dekamer$text, 
#'                        FUN = function(x) paste(x, collapse = " "))
#' 
#' idx <- sample(nrow(dekamer), size = round(nrow(dekamer) * 0.9))
#' traindata <- dekamer[idx, ]
#' testdata <- dekamer[-idx, ]
#' set.seed(123456789)
#' model <- embed_tagspace(x = traindata$text, 
#'                         y = traindata$question_theme_main, 
#'                         early_stopping = 0.8,
#'                         dim = 10, minCount = 5)
#' scores <- predict(model, testdata)                        
#' scores <- predict(model, testdata, type = "labels")
#' str(scores)
#' emb <- predict(model, testdata[, c("doc_id", "text")], type = "embedding")
#' knn <- predict(model, testdata[1:5, c("doc_id", "text")], type = "knn", k=3)
#' 
#' 
#' \dontrun{
#' library(udpipe)
#' data(dekamer, package = "ruimtehol")
#' dekamer <- subset(dekamer, question_theme_main == "DEFENSIEBELEID")
#' x <- udpipe(dekamer$question, "dutch", tagger = "none", parser = "none", trace = 100)
#' x <- x[, c("doc_id", "sentence_id", "sentence", "token")]
#' set.seed(123456789)
#' model <- embed_sentencespace(x, dim = 15, epoch = 5, minCount = 5)
#' scores <- predict(model, "Wat zijn de cijfers qua doorstroming van 2016?", 
#'                   basedoc = unique(x$sentence), k = 3) 
#' str(scores)
#' 
#' #' ## clean up for cran
#' file.remove(list.files(pattern = ".udpipe$"))
#' }
predict.textspace <- function(object, newdata, type = c("generic", "labels", "knn", "embedding"), 
                              k = 5L, sep = " ", basedoc, ...){
  type <- match.arg(type)
  if(is.data.frame(newdata)){
    stopifnot(all(c("doc_id", "text") %in% colnames(newdata)))
  }else{
    if(length(names(newdata)) == 0){
      newdata <- data.frame(doc_id = seq_along(newdata), text = newdata, stringsAsFactors = FALSE)  
    }else{
      newdata <- data.frame(doc_id = names(newdata), text = newdata, stringsAsFactors = FALSE)  
    }
  }
  if(type == "generic"){
    stopifnot(all(nchar(newdata$text) > 0))
    if(missing(basedoc)){
      basedoc <- as.character(c())
    }
    k <- as.integer(k)
    scores <- mapply(doc_id = newdata$doc_id, 
                     text = as.character(newdata$text), 
                     FUN=function(doc_id, text){
                       scores <- textspace_predict(object$model, input = text, sep = sep, k = k, basedoc = basedoc)
                       scores$doc_id <- doc_id
                       scores$text <- text
                       scores$prediction$label <- remove_label_prefix(object, scores$prediction$label_starspace)
                       scores$prediction <- scores$prediction[, c("label", "label_starspace", "similarity")]
                       scores[c("doc_id", "text", "prediction", "terms")]
                     }, SIMPLIFY = FALSE)
  }else if(type == "labels"){
    scores <- cbind_embedding_similarity(object, newdata = newdata, type = object$args$param$similarity)
  }else if(type == "knn"){
    scores <- mapply(doc_id = newdata$doc_id, 
                     text = as.character(newdata$text), 
                     FUN=function(doc_id, text){
                       scores <- starspace_knn(object, newdata = text, k = k)
                       scores <- scores$prediction
                       scores$doc_id <- rep(doc_id, nrow(scores))
                       scores[, c("doc_id", "label", "similarity", "rank")]
                     }, SIMPLIFY = FALSE)
  }else if(type == "embedding"){
    scores <- starspace_embedding(object, x = newdata$text, type = "document")
    rownames(scores) <- newdata$doc_id
  }
  scores
}



cbind_embedding_similarity <- function(object, newdata, type = c("cosine", "dot")) {
  type <- match.arg(type)
  stopifnot(inherits(object, "textspace"))
  stopifnot(is.data.frame(newdata))
  stopifnot(all(c("doc_id", "text") %in% colnames(newdata)))
  if(!requireNamespace("data.table", quietly = TRUE)){
    stop("cbind_embedding_similarity requires the data.table package, which you can install from cran with install.packages('data.table')")
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


#' @export
plot.textspace <- function(x, ...){
  if("iter" %in% names(x)){
    dataset <- data.frame(epoch = x$iter$epoch, error = x$iter$error, datatype = "training")
    if(length(x$iter$error_validation) > 0){
      dataset <- rbind(dataset, data.frame(epoch = x$iter$epoch, error = x$iter$error_validation, datatype = "validation"))
    }
    plot(error ~ epoch, data = dataset, type = "n", xlab = "Epoch", ylab = "Loss", ...)
    points(x = x$iter$epoch, y = x$iter$error, col = "steelblue", type = "b", pch = 20, lty = 1)
    if(length(x$iter$error_validation) > 0){
      points(x = x$iter$epoch, y = x$iter$error_validation, col = "purple", type = "b", lty = 2, pch = 20)
    }
    legend("topright", legend = c("Training", "Validation"),
           lty = 1:2, col = c("steelblue", "purple"),
           title = "Data")
  }else{
    stop("This is a model which was loaded using starspace_load_model, in which case plot.textspace does not work as the training error is not available.")
  }
}


#' @title K-nearest neighbours using a Starspace model 
#' @description K-nearest neighbours using a Starspace model 
#' @param object an object of class \code{textspace} as returned by \code{\link{starspace}} or \code{\link{starspace_load_model}}
#' @param newdata a character string of length 1
#' @param k integer with the number of nearest neighbours
#' @param ... not used
#' @export
#' @return a list with elements input and a data.frame called prediction which has columns called label, similarity and rank
starspace_knn <- function(object, newdata, k = 5, ...){
  stopifnot(inherits(object, "textspace"))
  stopifnot(is.character(newdata))
  stopifnot(length(newdata) == 1)
  stopifnot(nchar(newdata) > 0)
  k <- as.integer(k)
  knn <- textspace_knn(object$model, newdata, k)
  knn$prediction$rank <- seq_len(nrow(knn$prediction))
  knn
}

#' @title Load a Starspace model
#' @description Load a Starspace model
#' @param object the path to a Starspace model on disk
#' @param method character indicating the method of loading. Possible values are 'ruimtehol', 'binary' and 'tsv-data.table'. Defaults to 'ruimtehol'.
#' \itemize{
#' \item{method \code{'ruimtehol'} loads the model, embeddings and labels which were saved with saveRDS by calling \code{\link{starspace_save_model}} and re-initilises a new Starspace model with the embeddings and the same parameters used to build the model}
#' \item{method \code{'binary'} loads the embedding which were saved as a as a binary file using the original methods of the Starspace authors - see \code{\link{starspace_save_model}}}
#' \item{method \code{'tsv-data.table'} loads the embedding which were saved as a tab-delimited flat file using the fast data.table fread function - see \code{\link{starspace_save_model}}}
#' }
#' @param ... further arguments passed on to \code{\link{starspace}} in case of method 'tsv-data.table'
#' @export
#' @return an object of class textspace
#' @seealso \code{\link{starspace_save_model}}
#' @examples
#' data(dekamer, package = "ruimtehol")
#' dekamer$text <- strsplit(dekamer$question, "\\W")
#' dekamer$text <- lapply(dekamer$text, FUN = function(x) x[x != ""])
#' dekamer$text <- sapply(dekamer$text, 
#'                        FUN = function(x) paste(x, collapse = " "))
#' 
#' dekamer$target <- as.factor(dekamer$question_theme_main)
#' codes <- data.frame(code = seq_along(levels(dekamer$target)), 
#'                     label = levels(dekamer$target), stringsAsFactors = FALSE)
#' dekamer$target <- as.integer(dekamer$target)
#' set.seed(123456789)
#' model <- embed_tagspace(x = dekamer$text, 
#'                         y = dekamer$target, 
#'                         early_stopping = 0.8,
#'                         dim = 10, minCount = 5)
#' starspace_save_model(model, file = "textspace.ruimtehol", method = "ruimtehol",
#'                      labels = codes)
#' model <- starspace_load_model("textspace.ruimtehol", method = "ruimtehol")
#' 
#' 
#' ## clean up for cran
#' file.remove("textspace.ruimtehol")
starspace_load_model <- function(object, method = c("ruimtehol", "tsv-data.table", "binary"), ...){
  method <- match.arg(method)  
  stopifnot(is.character(object))
  stopifnot(file.exists(object))
  filename <- object
  if(method == "binary"){
    object <- textspace_load_model(filename, is_tsv = FALSE)
  }else if(method == "tsv-starspace"){
    object <- textspace_load_model(filename, is_tsv = TRUE)
  }else if(method == "tsv-data.table"){
    if(requireNamespace("data.table", quietly = TRUE)){
      x <- data.table::fread(filename, sep = "\t", encoding = "UTF-8")
      embeddings <- data.matrix(x[, -1, with = FALSE])
      rownames(embeddings) <- x$V1
      object <- starspace(embeddings = embeddings, ...)
    }else{
      stop("method tsv-data.table requires the data.table package, which you can install from cran with install.packages('data.table')")
    }
  }else if(method == "ruimtehol"){
    ruimte <- readRDS(filename)
    model <- ruimte$object
    model$args$data$testFile <- NULL
    arguments <- c(file = model$args$file, dim = model$args$dim, 
                   model$args$data, 
                   model$args$param, 
                   model$args$dictionary, 
                   model$args$options)
    arguments <- as.list(arguments)
    arguments$embeddings <- ruimte$embeddings
    arguments$embeddings_bucket_size <- 0L
    if("dictionary_size" %in% names(ruimte)){
      arguments$embeddings_bucket_size <- nrow(arguments$embeddings) - ruimte$dictionary_size
    }
    arguments$file <- NULL
    arguments$validationFile <- NULL
    object <- do.call(starspace, arguments)
    object$args$data$trainFile <- model$args$data$trainFile
    object$args$data$validationFile <- model$args$data$validationFile
    object$labels <- ruimte$labels
    if(!"label_starspace" %in% colnames(object$labels)){
      object$labels$label_starspace <- as.character(sapply(object$labels$code, FUN=function(code) paste(model$args$dictionary$label, code, sep = "")))  
    }
    object$iter <- model$iter
    for(att in setdiff(names(attributes(ruimte$object)), c("names", "class"))){
      attr(object, which = att) <- attr(ruimte$object, which = att)
    }
    object
  }
  class(object) <- "textspace"
  object
}

#' @title Save a starspace model as a binary or tab-delimited TSV file
#' @description Save a starspace model as a binary or a tab-delimited TSV file
#' @param object an object of class \code{textspace} as returned by \code{\link{starspace}} or \code{\link{starspace_load_model}}
#' @param file character string with the path to the file where to save the model
#' @param method character indicating the method of saving. Possible values are 'ruimtehol', 'binary', 'tsv-starspace' and 'tsv-data.table'. Defaults to 'ruimtehol'.
#' \itemize{
#' \item{The first method: \code{'ruimtehol'} saves the R object and the embeddings and optionally the label definitions with saveRDS. This object can be loaded back in with \code{\link{starspace_load_model}}.}
#' \item{The second method: \code{'tsv-data.table'} saves the model embeddings as a tab-delimited flat file using the fast data.table fwrite function}
#' \item{The third method: \code{'binary'} saves the model as a binary file using the original methods of the Starspace authors}
#' \item{The fourth method: \code{'tsv-starspace'} saves the model as a tab-delimited flat file using the original methods of the Starspace authors}
#' }
#' @param labels a data.frame with at least columns code and label which will be saved in case \code{method} is set to \code{'ruimtehol'}. 
#' This allows to store the mapping between Starspace labels and your own codes alongside the model, 
#' where code is your internal code and label is your label.\cr
#' A new column will be added to this data.frame called \code{label_starspace} which combines the 
#' Starspace prefix of the label with the code column of your provided data.frame, as this combination is the label starspace uses internally.
#' @export
#' @note It is advised to always use method 'ruimtehol' method as it works nicely together with the 
#' \code{\link{starspace_load_model}} function. It is the advised method unless you need to provide non-R users the models 
#' and you prefer using the methods provided by the Starspace authors instead of the faster and more portable 'ruimtehol' method.
#' @return invisibly, the character string with the file of the saved object
#' @seealso \code{\link{starspace_load_model}}
#' @examples
#' data(dekamer, package = "ruimtehol")
#' dekamer$text <- strsplit(dekamer$question, "\\W")
#' dekamer$text <- lapply(dekamer$text, FUN = function(x) x[x != ""])
#' dekamer$text <- sapply(dekamer$text, 
#'                        FUN = function(x) paste(x, collapse = " "))
#' 
#' dekamer$target <- as.factor(dekamer$question_theme_main)
#' codes <- data.frame(code = seq_along(levels(dekamer$target)), 
#'                     label = levels(dekamer$target), stringsAsFactors = FALSE)
#' dekamer$target <- as.integer(dekamer$target)
#' set.seed(123456789)
#' model <- embed_tagspace(x = dekamer$text, 
#'                         y = dekamer$target, 
#'                         early_stopping = 0.8,
#'                         dim = 10, minCount = 5)
#' starspace_save_model(model, file = "textspace.ruimtehol", method = "ruimtehol",
#'                      labels = codes)
#' model <- starspace_load_model("textspace.ruimtehol", method = "ruimtehol")
#' starspace_save_model(model, file = "embeddings.tsv", method = "tsv-data.table")
#' 
#' ## clean up for cran
#' file.remove("textspace.ruimtehol")
#' file.remove("embeddings.tsv")
starspace_save_model <- function(object, file = "textspace.ruimtehol",
                                 method = c("ruimtehol", "tsv-data.table", "binary", "tsv-starspace"),
                                 labels = data.frame(code = character(), label = character(), stringsAsFactors = FALSE)){
  stopifnot(inherits(object, "textspace"))
  method <- match.arg(method)
  if(method == "binary"){
    result <- textspace_save_model(object$model, file = file, as_tsv = FALSE)  
  }else if(method == "tsv-starspace"){
    if(!missing(file)){
      result <- textspace_save_model(object$model, file = file, as_tsv = TRUE)  
    }else{
      result <- textspace_save_model(object$model, file = object$args$file, as_tsv = TRUE)  
    }
  }else if(method == "tsv-data.table"){
    if(requireNamespace("data.table", quietly = TRUE)){
      ## Much quicker version of writing the model to a tsv
      embeddings <- as.matrix(object)
      x <- data.table::as.data.table(embeddings, keep.rownames = TRUE)
      data.table::fwrite(x, file = file, sep = "\t", col.names = FALSE)    
      result <- file
    }else{
      stop("method tsv-data.table requires the data.table package, which you can install from cran with install.packages('data.table')")
    }
  }else if(method == "ruimtehol"){
    stopifnot(inherits(labels, "data.frame"))
    stopifnot(all(c("code", "label") %in% colnames(labels)))
    labels$label_starspace <- as.character(sapply(labels$code, FUN=function(code) paste(object$args$dictionary$label, code, sep = "")))
    ## embeddings of LHS (words + labels + buckets in case ngram > 1), buckets at the end
    rn <- starspace_dictionary(object)$dictionary$term 
    ruimte <- list(
      object = object,
      labels = labels,
      dictionary_size = length(rn),
      embeddings = as.matrix(object, type = "LHS"))
    # ruimte <- list(
    #   object = object,
    #   labels = labels,
    #   embeddings = as.matrix(object, type = "all"))
    saveRDS(object = ruimte, file = file)
    result <- file
  }
  invisible(result)
}

#' @title Get the document or ngram embeddings
#' @description Get the document or ngram embeddings
#' @param object an object of class \code{textspace} as returned by \code{\link{starspace}} or \code{\link{starspace_load_model}}
#' @param x character vector with text to get the embeddings 
#' \itemize{
#' \item If \code{type} is set to 'document', will assume that a tab or a space is used as separator of each element of \code{x}.
#' \item If \code{type} is set to 'ngram', will assume that a space is used as separator of each element of \code{x}.
#' }
#' @param type the type of embedding requested. Either one of 'document' or 'ngram'. In case of document, 
#' the function returns the document embedding, in case of ngram the function returns the embedding of the 
#' provided ngram term. See the details section 
#' @details
#' \itemize{
#' \item{document embeddings look to the features (e.g. words) present in \code{x} and summate the embeddings of these to get a document embedding and 
#' divide this embedding by size^p in case dot similarity is used and the euclidean norm in case cosine similarity is used. 
#' Where size is the number of features (e.g. words) in \code{x}. 
#' If p=1, it's equivalent to taking average of embeddings while when p=0, it's equivalent to taking sum of embeddings. You can set p and similarity in \code{\link{starspace}} when you train the model.}
#' \item{for ngram embeddings, starspace is using a hashing trick to find out in which bucket the ngram lies and then retrieves the embedding of that. Note that if you specify ngram, 
#' you need to make sure \code{x} contains less features (e.g. words) then you've set \code{ngram} when you trained your model with \code{\link{starspace}}.}
#' }
#' @export
#' @return a matrix of embeddings
#' @examples 
#' data(dekamer, package = "ruimtehol")
#' dekamer$text <- strsplit(dekamer$question, "\\W")
#' dekamer$text <- lapply(dekamer$text, FUN = function(x) x[x != ""])
#' dekamer$text <- sapply(dekamer$text, 
#'                        FUN = function(x) paste(x, collapse = " "))
#' 
#' set.seed(123456789)
#' model <- embed_tagspace(x = tolower(dekamer$text), 
#'                         y = dekamer$question_theme_main, 
#'                         similarity = "dot",
#'                         early_stopping = 0.8, ngram = 1, p = 0.5,
#'                         dim = 10, minCount = 5)
#' embedding <- starspace_embedding(model, "federale politie", type = "document")
#' embedding_dictionary <- as.matrix(model)
#' embedding
#' colSums(embedding_dictionary[c("federale", "politie"), ]) / 2^0.5
#' 
#' \dontrun{
#' set.seed(123456789)
#' model <- embed_tagspace(x = tolower(dekamer$text), 
#'                         y = dekamer$question_theme_main, 
#'                         similarity = "cosine",
#'                         early_stopping = 0.8, ngram = 1, 
#'                         dim = 10, minCount = 5)
#' embedding <- starspace_embedding(model, "federale politie", type = "document")
#' embedding_dictionary <- as.matrix(model)
#' euclidean_norm <- function(x) sqrt(sum(x^2))
#' manual <- colSums(embedding_dictionary[c("federale", "politie"), ])
#' manual / euclidean_norm(manual)
#' embedding
#' 
#' set.seed(123456789)
#' model <- embed_tagspace(x = tolower(dekamer$text), 
#'                         y = dekamer$question_theme_main, 
#'                         similarity = "dot",
#'                         early_stopping = 0.8, ngram = 3, p = 0,
#'                         dim = 10, minCount = 5, bucket = 1)
#' starspace_embedding(model, "federale politie", type = "document")
#' starspace_embedding(model, "federale politie", type = "ngram")
#' }
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
as.matrix.textspace <- function(x, type = c("all", "labels", "words", "LHS", "RHS"), prefix = TRUE, ...){
  type <- match.arg(type)
  d <- starspace_dictionary(x)
  if("tsv" %in% names(list(...))){
    embedding_dimension <- x$args$dim
    filename <- tempfile()
    starspace_save_model(x, file = filename, method = "tsv-starspace")
    emb <- utils::read.delim(filename, header = FALSE, stringsAsFactors = FALSE, encoding = "UTF-8", colClasses = c("character", rep("numeric", embedding_dimension)))
    dn <- list(emb$V1, 1:(ncol(emb)-1))
    emb <- as.matrix(emb[, -1, drop = FALSE])
    dimnames(emb) <- dn  
  }else{
    if(type == "all"){
      emb <- starspace_embedding(object = x, x = d$dictionary$term, type = "ngram")    
    }else if(type == "labels"){
      if(length(d$labels) == 0){
        stop("You did not train the Starspace model with labels")
      }
      emb <- starspace_embedding(object = x, x = d$labels, type = "ngram")  
    }else if(type == "words"){
      words <- d$dictionary$term[d$dictionary$is_word]
      if(length(words) == 0){
        stop("Starspace model has no words, you must have trained it only with labels")
      }
      emb <- starspace_embedding(object = x, x = words, type = "ngram")  
    }else if(type %in% c("LHS", "RHS")){
      ## embeddings of LHS/RHS (words + labels + buckets in case ngram > 1), buckets at the end
      rn <- d$dictionary$term
      stopifnot(inherits(x, "textspace"))
      if(type == "LHS"){
        emb <- textspace_embedding_lhsrhs(x$model, type = "lhs")    
      }else if(type == "RHS"){
        emb <- textspace_embedding_lhsrhs(x$model, type = "rhs")  
      }
      rown <- seq_len(nrow(emb)) 
      rown[seq_along(rn)] <- rn
      rownames(emb) <- rown
    }
  }
  if(!prefix){
    idx <- which(rownames(emb) %in% d$labels)
    if(length(idx) > 0){
      rownames(emb)[idx] <- remove_label_prefix(x, rownames(emb)[idx])
    }
  }
  emb
}
