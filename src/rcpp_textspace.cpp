#include <Rcpp.h>
#include "starspace.h"

using namespace std;


// [[Rcpp::export]]
void textspace_help(std::string type = "help") {
  shared_ptr<starspace::Args> args = make_shared<starspace::Args>();
  if(type == "help"){
    args->printHelp();
  }else{
    args->printArgs();
  }
}

// [[Rcpp::export]]
Rcpp::List textspace_args(SEXP textspacemodel) {
  Rcpp::XPtr<starspace::StarSpace> sp(textspacemodel);
  // Get the list of model arguments
  std::shared_ptr<starspace::Args> args = sp->args_;
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("file") = args->model,
    Rcpp::Named("dim") = args->dim,
    Rcpp::Named("data") = Rcpp::List::create(
      Rcpp::Named("trainMode") = args->trainMode,
      Rcpp::Named("fileFormat") = args->fileFormat,
      Rcpp::Named("trainFile") = args->trainFile,
      Rcpp::Named("validationFile") = args->validationFile,
      Rcpp::Named("testFile") = args->testFile,
      Rcpp::Named("initModel") = args->initModel,
      Rcpp::Named("useWeight") = args->useWeight
    ),
    Rcpp::Named("param") = Rcpp::List::create(
      Rcpp::Named("loss") = args->loss,
      Rcpp::Named("margin") = args->margin,
      Rcpp::Named("similarity") = args->similarity,
      Rcpp::Named("epoch") = args->epoch,
      Rcpp::Named("adagrad") = args->adagrad,
      Rcpp::Named("lr") = args->lr,
      Rcpp::Named("termLr") = args->termLr,
      Rcpp::Named("norm") = args->norm,
      Rcpp::Named("maxNegSamples") = args->maxNegSamples,
      Rcpp::Named("negSearchLimit") = args->negSearchLimit,
      Rcpp::Named("p") = args->p,
      Rcpp::Named("shareEmb") = args->shareEmb,
      Rcpp::Named("ws") = args->ws,
      Rcpp::Named("dropoutLHS") = args->dropoutLHS,
      Rcpp::Named("dropoutRHS") = args->dropoutRHS,
      Rcpp::Named("initRandSd") = args->initRandSd
    ),
    Rcpp::Named("dictionary") = Rcpp::List::create(
      Rcpp::Named("minCount") = args->minCount,
      Rcpp::Named("minCountLabel") = args->minCountLabel,
      Rcpp::Named("ngrams") = args->ngrams,
      Rcpp::Named("bucket") = args->bucket,
      Rcpp::Named("label") = args->label
    ),
    Rcpp::Named("options") = Rcpp::List::create(
      Rcpp::Named("thread") = args->thread,
      Rcpp::Named("verbose") = args->verbose,
      Rcpp::Named("debug") = args->debug,
      Rcpp::Named("maxTrainTime") = args->maxTrainTime,
      Rcpp::Named("saveEveryEpoch") = args->saveEveryEpoch,
      Rcpp::Named("saveTempModel") = args->saveTempModel,
      Rcpp::Named("validationPatience") = args->validationPatience,
      Rcpp::Named("normalizeText") = args->normalizeText,
      Rcpp::Named("trainWord") = args->trainWord,
      Rcpp::Named("wordWeight") = args->wordWeight
    ),
    Rcpp::Named("test") = Rcpp::List::create(
      Rcpp::Named("basedoc") = args->basedoc,
      Rcpp::Named("predictionFile") = args->predictionFile,
      Rcpp::Named("K") = args->K,
      Rcpp::Named("excludeLHS") = args->excludeLHS),
      Rcpp::Named("isTrain") = args->isTrain
  );
  return out;
}



// [[Rcpp::export]]
Rcpp::List textspace(std::string model = "textspace.bin",
                     /* Arguments specific for training */
                     std::string trainFile = "",
                     std::string initModel = "",
                     std::string validationFile = "",
                     /* Arguments specific for test  */
                     std::string testFile = "",
                     std::string basedoc = "",
                     std::string predictionFile = "",
                     /* Rest are the starspace defaults from Starspace/src/utils/args */
                     std::string fileFormat = "fastText",
                     std::string label = "__label__",
                     std::string loss = "hinge",
                     std::string similarity = "cosine",
                     double lr = 0.01,
                     double termLr = 1e-9,
                     double norm = 1.0,
                     double margin = 0.05,
                     double initRandSd = 0.001,
                     double p = 0.5,
                     double dropoutLHS = 0.0,
                     double dropoutRHS = 0.0,
                     double wordWeight = 0.5,
                     size_t dim = 100,
                     int epoch = 5,
                     int ws = 5,
                     int maxTrainTime = 60*60*24*100,
                     int validationPatience = 10,
                     int thread = 10,
                     int maxNegSamples = 10,
                     int negSearchLimit = 50,
                     int minCount = 1,
                     int minCountLabel = 1,
                     int bucket = 2000000,
                     int ngrams = 1,
                     int trainMode = 0,
                     int K = 5,
                     bool verbose = false,
                     bool debug = false,
                     bool adagrad = true,
                     bool normalizeText = false,
                     bool saveEveryEpoch = false,
                     bool saveTempModel = false,
                     bool shareEmb = true,
                     bool useWeight = false,
                     bool trainWord = false,
                     bool excludeLHS = false) {
  shared_ptr<starspace::Args> args = make_shared<starspace::Args>();
  args->model = model;
  /*
   * Check if it is training or testing
   */
  if(trainFile == "" && testFile == ""){
    Rcpp::stop("Either provide a training file or a test file");
  }
  if(trainFile != "" && testFile != ""){
    Rcpp::stop("Either provide a training file or a test file, not both");
  }
  if(std::ifstream(trainFile)){
    args->isTrain = true;  
    args->trainFile = trainFile;
    if(std::ifstream(initModel))      args->initModel = initModel;
    if(std::ifstream(validationFile)) args->validationFile = validationFile;
  }else if(std::ifstream(testFile)){
    args->isTrain = false;  
    args->testFile = testFile;
    if(std::ifstream(basedoc))        args->basedoc = basedoc;
    if(std::ifstream(predictionFile)) args->predictionFile = predictionFile;
  }else{
    Rcpp::stop("No valid trainFile nor testFile. Please check your path and check if the file is not opened.");
  }
  /*
   * Assign the other parameters of the modelling
   */
  args->fileFormat = fileFormat;
  args->label = label;
  args->loss = loss;
  args->similarity = similarity;
  args->lr = lr;
  args->termLr = termLr;
  args->norm = norm;
  args->margin = margin;;
  args->initRandSd = initRandSd;
  args->p = p;
  args->dropoutLHS = dropoutLHS;
  args->dropoutRHS = dropoutRHS;
  args->wordWeight = wordWeight;
  args->dim = dim;
  args->epoch = epoch;
  args->ws = ws;
  args->maxTrainTime = maxTrainTime;
  args->validationPatience = validationPatience;
  args->thread = thread;
  args->maxNegSamples = maxNegSamples;
  args->negSearchLimit = negSearchLimit;
  args->minCount = minCount;
  args->minCountLabel = minCountLabel;
  args->bucket = bucket;
  args->ngrams = ngrams;
  args->trainMode = trainMode;
  args->K = K;
  args->verbose = verbose;
  args->debug = debug;
  args->adagrad = adagrad;
  args->normalizeText = normalizeText;
  args->saveEveryEpoch = saveEveryEpoch;
  args->saveTempModel = saveTempModel;
  args->shareEmb = shareEmb;
  args->useWeight = useWeight;
  args->trainWord = trainWord;
  args->excludeLHS = excludeLHS;

  /*
   * Build and save the model
   */
  Rcpp::XPtr<starspace::StarSpace> sp(new starspace::StarSpace(args), true);
  Rcpp::List out;
  if(args->isTrain){
    if(std::ifstream(args->initModel)){
      sp->initFromSavedModel(args->initModel);
    }else{
      sp->init();  
    }
    sp->train();
    sp->saveModel(args->model);  
    out = Rcpp::List::create(
      Rcpp::Named("model") = sp,
      Rcpp::Named("args") = textspace_args(sp));
  }else{
    sp->initFromSavedModel(args->model);
    sp->evaluate();
    out = Rcpp::List::create(
      Rcpp::Named("model") = sp,
      Rcpp::Named("args") = textspace_args(sp),
      Rcpp::Named("test") = "UNDER CONSTRUCTION: capture results of sp->evaluate() or write own sp->evaluate");
  }

  /*
   * Return pointer to the model and the used arguments as a list
   */
  return out;
}


// [[Rcpp::export]]
Rcpp::List textspace_load_model(const std::string file_model, bool is_tsv = false) {
  shared_ptr<starspace::Args> args = make_shared<starspace::Args>();
  args->model = file_model;
  Rcpp::XPtr<starspace::StarSpace> sp(new starspace::StarSpace(args), true);
  if(is_tsv){
    sp->initFromTsv(args->model);
  }else{
    sp->initFromSavedModel(args->model);  
  }
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("model") = sp,
    Rcpp::Named("args") = textspace_args(sp));
  return out;
}

// [[Rcpp::export]]
Rcpp::List textspace_save_model(SEXP textspacemodel, std::string file = "textspace.tsv") {
  Rcpp::XPtr<starspace::StarSpace> sp(textspacemodel);
  sp->saveModelTsv(file);
  return file;
}


// [[Rcpp::export]]
Rcpp::List textspace_dictionary(SEXP textspacemodel) {
  Rcpp::XPtr<starspace::StarSpace> sp(textspacemodel);
  std::vector<std::string> labels;
  std::vector<std::string> key;
  std::vector<bool> is_word;
  std::vector<bool> is_label;
  starspace::entry_type wordorlabel;
  
  for(int32_t i = 0; i < sp->dict_->size(); i++){
    key.push_back(sp->dict_->getSymbol(i));
    wordorlabel = sp->dict_->getType(i);
    is_word.push_back(wordorlabel == starspace::entry_type::word);
    is_label.push_back(wordorlabel == starspace::entry_type::label);
  }
  for(int32_t i = 0; i < sp->dict_->nlabels(); i++){
    labels.push_back(sp->dict_->getLabel(i));
  }
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("ntokens") = sp->dict_->ntokens(),
    Rcpp::Named("nwords") = sp->dict_->nwords(),
    Rcpp::Named("nlabels") = sp->dict_->nlabels(),
    Rcpp::Named("labels") = labels,
    Rcpp::Named("dictionary_size") = sp->dict_->size(),
    Rcpp::Named("dictionary") = Rcpp::DataFrame::create(
      Rcpp::Named("term") = key,
      Rcpp::Named("is_word") = is_word,
      Rcpp::Named("is_label") = is_label,
      Rcpp::Named("stringsAsFactors") = false)
  );
  return out;
}


// [[Rcpp::export]]
Rcpp::NumericMatrix textspace_embedding_doc(SEXP textspacemodel, Rcpp::StringVector x) {
  Rcpp::XPtr<starspace::StarSpace> sp(textspacemodel);
  // set useWeight by default. use 1.0 for default weight if weight is not found
  sp->args_->useWeight = true;
  // get docvector of each document and return it as a matrix
  Rcpp::NumericMatrix embedding(x.size(), sp->args_->dim);
  rownames(embedding) = x;
  for (int i = 0; i < x.size(); i++){
    std::string input = Rcpp::as<std::string>(x[i]);
    starspace::Matrix<starspace::Real> vec = sp->getDocVector(input, " \t");
    if(vec.numRows() > 1){
      Rcpp::stop("Unexpected outcome of sp->getDocVector, please report to the ruimtehol maintainer.");
    }
    for(int j = 0; j < vec.numCols(); j++){
      embedding(i, j) = vec.cell(0, j);
    }
  }
  return embedding;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix textspace_embedding_ngram(SEXP textspacemodel, Rcpp::StringVector x) {
  Rcpp::XPtr<starspace::StarSpace> sp(textspacemodel);
  Rcpp::NumericMatrix embedding(x.size(), sp->args_->dim);
  rownames(embedding) = x; 
  for (int i = 0; i < x.size(); i++){
    std::string input = Rcpp::as<std::string>(x[i]);
    starspace::MatrixRow vec = sp->getNgramVector(input);
    for(int j = 0; j < vec.size(); j++){
      embedding(i, j) = vec[j];
    }
  }
  return embedding;
}

// [[Rcpp::export]]
Rcpp::List textspace_predict(SEXP textspacemodel, std::string input, Rcpp::StringVector basedoc = "", std::string sep = " ") { 
  Rcpp::XPtr<starspace::StarSpace> sp(textspacemodel);
  // Set dropout probability to 0 in test case.
  sp->args_->dropoutLHS = 0.0;
  sp->args_->dropoutRHS = 0.0;
  bool user_basedocs = false;
  // Load set of possible things to predict (basedoc), either from file or as a character vector of possible labels or take the labels from the dictionary
  sp->baseDocs_.clear();
  sp->baseDocVectors_.clear();
  if(basedoc.size() > 0){
    std::string file = Rcpp::as<std::string>(basedoc[0]);
    if(basedoc.size() == 1 && file != "" && std::ifstream(file)){
      // basedoc is a file
      sp->args_->basedoc = file;
      sp->args_->fileFormat = "labelDoc";
      sp->loadBaseDocs();
    }else{
      // basedoc is a character vector
      for (int i = 0; i < basedoc.size(); i++){
        std::string line = Rcpp::as<std::string>(basedoc[i]);
        vector<starspace::Base> ids;
        sp->parseDoc(line, ids, "\t ");
        sp->baseDocs_.push_back(ids);
        sp->baseDocVectors_.push_back(sp->model_->projectRHS(ids));
      }
      user_basedocs = true;
    }
  }else{
    sp->loadBaseDocs();
  }
  /*
  cout << sp->baseDocs_.size() << endl;
  for(int i=0; i < sp->baseDocs_.size(); i++){
    cout << sp->dict_->getSymbol(sp->baseDocs_[i][0].first) << endl;
  }
  */

  // Do the prediction
  std::vector<starspace::Base> query_vec;
  std::vector<starspace::Predictions> predictions;
  std::vector<starspace::Base> tokens;
  // split according to separator of the input query and put in query_vec all tokens which are part of the dictionary only
  sp->parseDoc(input, query_vec, sep);
  sp->predictOne(query_vec, predictions);
  std::vector<std::string> label;
  std::vector<std::string> basedoc_terms;
  std::vector<int32_t> basedoc_index;
  std::vector<float> prob;
  for (int i = 0; i < predictions.size(); i++) {
    prob.push_back(predictions[i].first);
    basedoc_index.push_back(predictions[i].second + 1); 
    tokens = sp->baseDocs_[predictions[i].second]; 
    std::string tokensline = "";
    if(user_basedocs){
      // If basedoc is given by R user as a character vector, should return just that instead of all the terms which are in the dictionary and are part of the basedoc
      label.push_back(Rcpp::as<std::string>(basedoc[predictions[i].second]));
      for (auto t : tokens) {
        if (t.first < sp->dict_->size()) {
          if(tokensline == ""){
            tokensline = tokensline + sp->dict_->getSymbol(t.first);
          }else{
            tokensline = tokensline + sep + sp->dict_->getSymbol(t.first);  
          }
        }
      }
      basedoc_terms.push_back(tokensline);
    }else{
      for (auto t : tokens) {
        if (t.first < sp->dict_->size()) {
          label.push_back(sp->dict_->getSymbol(t.first));
        }
      }
    }
  }
  Rcpp::List out = Rcpp::List::create(Rcpp::Named("input") = input, 
                                      Rcpp::Named("prediction") = Rcpp::DataFrame::create(
                                        Rcpp::Named("label") = label,
                                        Rcpp::Named("similarity") = prob,
                                        Rcpp::Named("stringsAsFactors") = false),
                                      Rcpp::Named("terms") = Rcpp::List::create(
                                        Rcpp::Named("basedoc_index") = basedoc_index,
                                        Rcpp::Named("basedoc_terms") = basedoc_terms)); 
  return out;
}

// [[Rcpp::export]]
Rcpp::List textspace_knn(SEXP textspacemodel, const std::string line, int k) {
  Rcpp::XPtr<starspace::StarSpace> sp(textspacemodel);
  starspace::Matrix<starspace::Real> vec = sp->getDocVector(line, " ");
  /*
  for(int i = 0; i < vec.numRows(); i++){
    for(int j = 0; j < vec.numCols(); j++){
      cout << vec.cell(i,j) << ' ';    
    }  
   cout << endl;    
  }
  */
  std::vector<std::pair<int32_t, starspace::Real>> preds = sp->model_->findLHSLike(vec, k);
  std::vector<std::string> label;
  std::vector<float> prob;
  for (auto n : preds) {
    label.push_back(sp->dict_->getSymbol(n.first));
    prob.push_back(n.second);
  }
  Rcpp::List out = Rcpp::List::create(Rcpp::Named("input") = line, 
                                      Rcpp::Named("prediction") = Rcpp::DataFrame::create(
                                        Rcpp::Named("label") = label,
                                        Rcpp::Named("similarity") = prob,
                                        Rcpp::Named("stringsAsFactors") = false)); 
  return out;
}


