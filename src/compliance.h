
#pragma once
#include <Rcpp.h>
#include <stdlib.h>

#define exit(EXIT_FAILURE) Rcpp::stop("Incorrect Starspace usage")
#define abort()            Rcpp::stop("Incorrect Starspace usage")

/*
#define cerr Rcerr 
#define cout Rcout

int rand();
void srand(unsigned int seed);

namespace std {
  extern std::ostream Rcout;
  extern std::ostream Rcerr;
}
*/