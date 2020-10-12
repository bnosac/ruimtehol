#' @title Dataset from 2017 with Questions and Answers in the Belgium Federal Parliament
#' @description Dataset from 2017 with Questions asked by members of the Belgian Federal Parliament 
#' and the Answers provided to these questions.\cr
#' The dataset was extracted from \url{http://data.dekamer.be} and contains questions asked by persons in the Belgium Federal parliament
#' and answers given by the departments of the Federal Belgian Ministers. \cr
#' The language of this dataset provided in this R package has been restricted to Dutch. \cr
#' 
#' The dataset contains the following information: 
#' \itemize{
#' \item doc_id: a unique identifier
#' \item depotdat: the date when the question was registered
#' \item aut_party / aut_person / aut_language: who asked the question and which political party is he/she a member of + the language of the person who asked the question
#' \item question: the question itself (always in Dutch)
#' \item question_theme_main: the main theme of the question
#' \item question_theme: a comma-separated list of all themes the question is about
#' \item answer: the answer given by the department of the minister (always in Dutch)
#' \item answer_deptpres, answer_department, answer_subdepartment: to which ministerial department has the question been raised to and answered by
#' }
#' @name dekamer
#' @docType data
#' @source \url{http://data.dekamer.be}, data is provided by www.dekamer.be in the public domain (CC0).
#' @examples
#' data(dekamer)
#' str(dekamer)
NULL


#' @title Dataset containing relevant terminology for each theme of the \code{dekamer} dataset
#' @description Dataset containing relevant terminology for each theme of the \code{\link{dekamer}} dataset
#' 
#' The dataset contains the following information: 
#' \itemize{
#' \item theme: a theme, corresponding to the \code{question_theme_main} field in the \code{\link{dekamer}} dataset
#' \item term: a word which describes the \code{theme}
#' \item n: a measure of information indicating how relevant the term is (frequency of occurrence)
#' }
#' @name dekamer_theme_terminology
#' @docType data
#' @examples
#' data(dekamer_theme_terminology)
#' str(dekamer_theme_terminology)
NULL

