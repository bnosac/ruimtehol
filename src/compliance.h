#pragma once
#include <Rcpp.h>

#define exit(EXIT_FAILURE) Rcpp::stop("Incorrect Starspace usage")
#define abort()            Rcpp::stop("Incorrect Starspace usage")
/*
#define cerr Rcout 
#define cout Rcout
*/
int rand();
void srand(unsigned int seed);

namespace std {
  extern std::ostream Rcout;
}
