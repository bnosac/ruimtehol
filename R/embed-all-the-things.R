
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
#' @param thread integer with the number of threads to use. Defaults to 10.
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
#' \item bucket:          number of buckets [2000000]
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
#' data(dekamer, package = "ruimtehol")
#' dekamer$text <- gsub("\\.([[:digit:]]+)\\.", ". \\1.", x = dekamer$question)
#' x <- strsplit(dekamer$text, "\\W")
#' x <- lapply(x, FUN = function(x) setdiff(x, ""))
#' x <- sapply(x, FUN = function(x) paste(x, collapse = " "))
#' 
#' \dontrun{
#' idx <- sample.int(n = nrow(dekamer), size = 6000)
#' writeLines(x[idx], con = "traindata.txt")
#' writeLines(x[-idx], con = "validationdata.txt")
#' 
#' m <- starspace(model = "mymodel.bin", 
#'                file = "traindata.txt", validationFile = "validationdata.txt", 
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
                      thread = 10, ...) {
  ldots <- list(...)
  if(!"embeddings" %in% names(ldots)){
    file <- path.expand(file)  
  }else{
    file <- ""
  }
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
starspace_dictionary <- function(object){
  stopifnot(inherits(object, "textspace"))
  textspace_dictionary(object$model)
}


#' @title Predict using a Starspace model 
#' @description Predict using a Starspace model 
#' @param object an object of class \code{textspace} as returned by \code{\link{starspace}} or \code{\link{starspace_load_model}}
#' @param newdata a character string of length 1
#' @param k integer with the number of predictions to make. Defaults to 5.
#' @param sep character string used to split \code{newdata} using boost::split
#' @param basedoc optional, either a character vector of possible elements to predict or 
#' the path to a file in labelDoc format, containing basedocs which are set of possible things to predict, if different than 
#' the ones from the training data
#' @param ... not used
#' @export
#' @return a list with elements 
#' \enumerate{
#' \item input: the character string passed on to newdata
#' \item prediction: data.frame called prediction which has columns called label, label_starspace and similarity indicating the predicted label and the similarity of the input ot the label
#' \item terms: a list with elements basedoc_index and basedoc_terms indicating the position in basedoc and the terms which are part of the dictionary which are used to find the similarity
#' }
predict.textspace <- function(object, newdata, k = 5L, sep = " ", basedoc, ...){
  stopifnot(is.character(newdata))
  stopifnot(length(newdata) == 1)
  stopifnot(nchar(newdata) > 0)
  if(missing(basedoc)){
    capture.output(scores <- textspace_predict(object$model, input = newdata, sep = sep, k = as.integer(k), basedoc = as.character(c())))
  }else{
    capture.output(scores <- textspace_predict(object$model, input = newdata, sep = sep, k = as.integer(k), basedoc = basedoc))
  }
  length_label_prefix <- nchar(object$args$dictionary$label)
  scores$prediction$label <- substr(scores$prediction$label_starspace, length_label_prefix + 1L, nchar(scores$prediction$label_starspace))
  scores$prediction <- scores$prediction[, c("label", "label_starspace", "similarity")]
  scores
}

#' @export
plot.textspace <- function(x, ...){
  if("iter" %in% names(x)){
    dataset <- data.frame(epoch = x$iter$epoch, error = x$iter$error, datatype = "training")
    if(length(x$iter$error_validation) > 0){
      dataset <- rbind(dataset, data.frame(epoch = x$iter$epoch, error = x$iter$error_validation, datatype = "validation"))
    }
    plot(error ~ epoch, data = dataset, type = "n", xlab = "Epoch", ylab = "Error", ...)
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
#' @return a list with elements input and a data.frame called prediction which has columns called label and similarity
starspace_knn <- function(object, newdata, k = 5, ...){
  stopifnot(inherits(object, "textspace"))
  stopifnot(is.character(newdata))
  stopifnot(length(newdata) == 1)
  stopifnot(nchar(newdata) > 0)
  k <- as.integer(k)
  textspace_knn(object$model, newdata, k)
}

#' @title Load a Starspace model
#' @description Load a Starspace model
#' @param object either the path to a Starspace model on disk or an object of class \code{textspace} which you want to reload.
#' @param method character indicating the method of loading. Possible values are 'binary', 'tsv-starspace', 'tsv-data.table'. Defaults to 'binary'.
#' \enumerate{
#' \item{The first method: \code{'binary'} loads the model which was saved a binary file using the Starspace methods - see \code{\link{starspace_save_model}}}
#' \item{The second method: \code{'tsv-starspace'} loads the model which was saved as a tab-delimited flat file using the Starspace methods - see \code{\link{starspace_save_model}}}
#' \item{The third method: \code{'tsv-data.table'} loads the model which was saved as a tab-delimited flat file using the fast data.table fread function - see \code{\link{starspace_save_model}}}
#' \item{The fourth method: \code{'ruimtehol'} loads the model, embeddings and labels which were saved with saveRDS by calling \code{\link{starspace_save_model}} and re-initilises a new Starspace model with the embeddings and the same parameters used to build the model}
#' }
#' @param ... further arguments passed on to \code{\link{starspace}} in case of method 'tsv-data.table'
#' @export
#' @return an object of class textspace
#' @seealso \code{\link{starspace_save_model}}
starspace_load_model <- function(object, method = c("binary", "tsv-starspace", "tsv-data.table", "ruimtehol"), ...){
  method <- match.arg(method)  
  if(inherits(object, "textspace")){
    filename <- object$args$file
    object <- textspace_load_model(filename, is_tsv = FALSE)
  }else{
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
                     model$args$data, model$args$param, model$args$dictionary, model$args$options)
      arguments <- as.list(arguments)
      arguments$embeddings <- ruimte$embeddings
      object <- do.call(starspace, arguments)
      object$labels <- ruimte$labels
      if(!"label_starspace" %in% colnames(object$labels)){
        object$labels$label_starspace <- as.character(sapply(object$labels$code, FUN=function(code) paste(model$args$dictionary$label, code, sep = "")))  
      }
      object$iter <- model$iter
      object
    }
  }
  class(object) <- "textspace"
  object
}

#' @title Save a starspace model as a binary or tab-delimited TSV file
#' @description Save a starspace model as a binary or a tab-delimited TSV file
#' @param object an object of class \code{textspace} as returned by \code{\link{starspace}} or \code{\link{starspace_load_model}}
#' @param file character string with the path to the file where to save the model, in case as_tsv is set to \code{TRUE}
#' @param method character indicating the method of saving. Possible values are 'binary', 'tsv-starspace', 'tsv-data.table'. Defaults to 'binary'.
#' \enumerate{
#' \item{The first method: \code{'binary'} saves the model as a binary file using the Starspace methods}
#' \item{The second method: \code{'tsv-starspace'} saves the model as a tab-delimited flat file using the Starspace methods}
#' \item{The third method: \code{'tsv-data.table'} saves the model as a tab-delimited flat file using the fast data.table fread function}
#' \item{The fourth method: \code{'ruimtehol'} saves the R object and the embeddings and optionally the label definitions with saveRDS}
#' }
#' @param labels a data.frame with at least columns code and label which will be saved in case \code{method} is \code{'ruimtehol'}. Allowing you to
#' save the label identifier. Internally a new column will be added to this data.frame called \code{label_starspace} which combines the 
#' Starspace prefix of the label with the code column of your provided data.frame.
#' This allows to store the mapping between Starspace labels and your own codes.
#' @export
#' @return invisibly, the character string with the file of the saved object
#' @seealso \code{\link{starspace_load_model}}
#' @examples
#' data(dekamer, package = "ruimtehol")
#' dekamer$text <- gsub("\\.([[:digit:]]+)\\.", ". \\1.", x = dekamer$question)
#' dekamer$text <- strsplit(dekamer$text, "\\W")
#' dekamer$text <- lapply(dekamer$text, FUN = function(x) setdiff(x, ""))
#' dekamer$text <- sapply(dekamer$text, 
#'                        FUN = function(x) paste(x, collapse = " "))
#' 
#' dekamer$target <- as.factor(dekamer$question_theme_main)
#' codes <- data.frame(code = seq_along(levels(dekamer$target)), 
#'                     label = levels(dekamer$target), stringsAsFactors = FALSE)
#' dekamer$target <- as.integer(dekamer$target)
#' model <- embed_tagspace(x = dekamer$text, 
#'                         y = dekamer$target, 
#'                         early_stopping = 0.8,
#'                         dim = 10, minCount = 5)
#' starspace_save_model(model, file = "textspace.ruimtehol", method = "ruimtehol",
#'                      labels = codes)
#' model <- starspace_load_model("textspace.ruimtehol", method = "ruimtehol")
#' file.remove("textspace.ruimtehol")
starspace_save_model <- function(object, file = "textspace.tsv",
                                 method = c("binary", "tsv-starspace", "tsv-data.table", "ruimtehol"),
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
    ruimte <- list(
      object = object,
      labels = labels,
      embeddings = as.matrix(object))
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
#' \item If \code{type} is set to 'document', will assume that a space followed by a tab is used as separator of the sentences of each element of \code{x}.
#' \item If \code{type} is set to 'ngram', will assume that a space is used as separator of the words in case \code{x} contains words.
#' }
#' @param type the type of embedding requested. Either one of 'document' or 'ngram'. In case of document, 
#' the function returns the document embedding, in case of ngram the function returns the embedding of the 
#' provided ngram term which is used in the model building. 
#' @export
#' @return a matrix of embeddings
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
as.matrix.textspace <- function(x, type = c("all", "labels", "words"), ...){
  type <- match.arg(type)
  if("tsv" %in% names(list(...))){
    embedding_dimension <- x$args$dim
    filename <- tempfile()
    starspace_save_model(x, file = filename, method = "tsv-starspace")
    x <- utils::read.delim(filename, header = FALSE, stringsAsFactors = FALSE, encoding = "UTF-8", colClasses = c("character", rep("numeric", embedding_dimension)))
    dn <- list(x$V1, 1:(ncol(x)-1))
    x <- as.matrix(x[, -1, drop = FALSE])
    dimnames(x) <- dn  
  }else{
    d <- starspace_dictionary(x)
    if(type == "all"){
      x <- starspace_embedding(object = x, x = d$dictionary$term, type = "ngram")    
    }else if(type == "labels"){
      if(length(d$labels) == 0){
        stop("You did not train the Starspace model with labels")
      }
      x <- starspace_embedding(object = x, x = d$labels, type = "ngram")  
    }else if(type == "words"){
      words <- d$dictionary$term[d$dictionary$is_word]
      if(length(words) == 0){
        stop("Starspace model has no words, you must have trained it only with labels")
      }
      x <- starspace_embedding(object = x, x = words, type = "ngram")  
    }
  }
  x
}
