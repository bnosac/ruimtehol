/*
#include <Rcpp.h>
#include <stdlib.h>

// [[Rcpp::export]]
int rand() {
  int r = R::unif_rand() * RAND_MAX;
  return r;
}
 
// [[Rcpp::export]]
void srand(unsigned int seed){
  Rcpp::Environment base("package:base"); 
  Rcpp::Function set_seed = base["set.seed"];    
  set_seed(seed);
};

namespace std {
  std::ostream Rcout(Rcpp::Rcout.rdbuf());
  std::ostream Rcerr(Rcpp::Rcerr.rdbuf());
}
*/
