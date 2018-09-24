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
#' \item question_theme: a pipe-separated list of all themes the question is about
#' \item answer: the answer given by the department of the minister (always in Dutch)
#' \item answer_deptpres, answer_department, answer_subdepartment: to which ministerial department has the question been raised to and answered by
#' }
#' @name dekamer
#' @docType data
#' @source \url{http://data.dekamer.be}, data is provided by \url{http://www.dekamer.be} in the public domain (CC0).
#' @examples
#' data(dekamer)
#' str(dekamer)
NULL
