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
      //Rcpp::Named("batchSize") = args->batchSize,
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

Rcpp::List textspace_train(SEXP textspacemodel) {
  Rcpp::XPtr<starspace::StarSpace> sp(textspacemodel);
  std::vector<int> train_epoch;
  std::vector<float> train_rate;
  std::vector<float> train_error;
  std::vector<float> validation_error;
  
  float rate = sp->args_->lr;
  float decrPerEpoch = (rate - 1e-9) / sp->args_->epoch;
  
  int impatience = 0;
  float best_valid_err = 1e9;
  auto t_start = std::chrono::high_resolution_clock::now();
  Rcpp::Function format_posixct("format.POSIXct");
  Rcpp::Function sys_time("Sys.time");
  
  // Also collecting error at start before training (added with respect to Starspace code)
  Rcpp::Rcout << Rcpp::as<std::string>(format_posixct(sys_time())) << " Initialising with learning rate " << rate << endl;
  train_epoch.push_back(0);
  train_rate.push_back(rate);
  auto err = sp->model_->test(sp->trainData_, sp->args_->thread);
  train_error.push_back(err);
  Rcpp::Rcout << "                     > Training data error   " << err << endl;
  if (sp->validData_ != nullptr) {
    auto valid_err = sp->model_->test(sp->validData_, sp->args_->thread);
    validation_error.push_back(valid_err);
    Rcpp::Rcout << "                     > Validation data error " << valid_err << endl;
  }
  // Done collecting error at start before training (added with respect to Starspace code)
  
  for (int i = 0; i < sp->args_->epoch; i++) {
    if (sp->args_->saveEveryEpoch && i > 0) {
      auto filename = sp->args_->model;
      if (sp->args_->saveTempModel) {
        filename = filename + "_epoch" + std::to_string(i);
      }
      sp->saveModel(filename);
      sp->saveModelTsv(filename + ".tsv");
    }
    Rcpp::Rcout << Rcpp::as<std::string>(format_posixct(sys_time())) << " Start training epoch " << i+1 << " with learning rate " << rate << endl;
    auto err = sp->model_->train(sp->trainData_, sp->args_->thread,
                                 t_start,  i,
                                 rate, rate - decrPerEpoch);
    train_epoch.push_back(i + 1);
    train_rate.push_back(rate);
    train_error.push_back(err);
    /*
    Rprintf("\n ---+++ %20s %4d Train error : %3.8f +++--- %c%c%c\n",
            "Epoch", i, err,
            0xe2, 0x98, 0x83);
     */
    Rcpp::Rcout << "                     > Training data error   " << err << endl;
    if (sp->validData_ != nullptr) {
      auto valid_err = sp->model_->test(sp->validData_, sp->args_->thread);
      validation_error.push_back(valid_err);
      Rcpp::Rcout << "                     > Validation data error " << valid_err << endl;
      if (valid_err > best_valid_err) {
        impatience += 1;
        if (impatience > sp->args_->validationPatience) {
          Rcpp::Rcout << "Ran out of Patience! Early stopping based on validation set." << endl;
          break;
        }
      } else {
        best_valid_err = valid_err;
      }
    }
    rate -= decrPerEpoch;
    
    auto t_end = std::chrono::high_resolution_clock::now();
    auto tot_spent = std::chrono::duration<double>(t_end-t_start).count();
    if (tot_spent > sp->args_->maxTrainTime) {
      Rcpp::Rcout << "MaxTrainTime exceeded." << endl;
      break;
    }
    Rcpp::checkUserInterrupt();
  }
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("epoch") = train_epoch,
    Rcpp::Named("lr") = train_rate,
    Rcpp::Named("error") = train_error,
    Rcpp::Named("error_validation") = validation_error
  );
  return out;
}


// [[Rcpp::export]]
Rcpp::List textspace(std::string model = "textspace.bin",
                     bool save = false,
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
                     int thread = 1,
                     int maxNegSamples = 10,
                     int negSearchLimit = 50,
                     int minCount = 1,
                     int minCountLabel = 1,
                     int bucket = 100000,
                     int ngrams = 1,
                     int trainMode = 0,
                     int K = 5,
                     int batchSize = 5,
                     bool verbose = false,
                     bool debug = false,
                     bool adagrad = true,
                     bool normalizeText = false,
                     bool saveEveryEpoch = false,
                     bool saveTempModel = false,
                     bool shareEmb = true,
                     bool useWeight = false,
                     bool trainWord = false,
                     bool excludeLHS = false,
                     Rcpp::NumericMatrix embeddings = Rcpp::NumericMatrix(0, 100),
                     int embeddings_bucket_size = 0) {
  shared_ptr<starspace::Args> args = make_shared<starspace::Args>();
  args->model = model;
  /*
   * Check if it is training or testing
   */
  bool load_from_r = false;
  if(embeddings.nrow() > 0){
    load_from_r = true;
    dim = embeddings.ncol();
  }
  if(trainFile == "" && testFile == "" && !load_from_r){
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
  }else if(load_from_r){
    args->isTrain = false;  
    //if(std::ifstream(initModel))      args->initModel = initModel;
    if(std::ifstream(trainFile)){
      args->isTrain = true;  
      args->trainFile = trainFile; 
    }
    if(std::ifstream(validationFile)) args->validationFile = validationFile;
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
  //args->batchSize = batchSize;
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
  if(load_from_r){
    Rcpp::List dimnames = embeddings.attr("dimnames");
    Rcpp::CharacterVector terminology = dimnames[0];
    //Rcpp::Rcout << "Set up dictionary" << endl;
    sp->dict_ = make_shared<starspace::Dictionary>(sp->args_);
    // embeddings of hashed buckets come after the dictionary
    for (int i = 0; i < (terminology.size() - embeddings_bucket_size); i++){
      std::string symbol = Rcpp::as<std::string>(terminology[i]);
      sp->dict_->insert(symbol);
    }
    sp->dict_->computeCounts();
    //Rcpp::Rcout << "Load embedding model" << endl;
    sp->model_ = make_shared<starspace::EmbedModel>(sp->args_, sp->dict_);
    for (int i = 0; i < terminology.size(); i++){
      if(i < (int)(sp->model_->LHSEmbeddings_->numRows())){
        if(i < (terminology.size() - embeddings_bucket_size)){
          // embeddings of real dictionary
          std::string symbol = Rcpp::as<std::string>(terminology[i]);
          auto idx = sp->dict_->getId(symbol);  
          if (idx == -1) {
            Rcpp::Rcout << "Failed to insert embedding for term " << symbol << endl;
          }else{
            auto row = sp->model_->LHSEmbeddings_->row(idx);
            for (unsigned int j = 0; j < args->dim; j++) {
              row(j) = (float)(embeddings(i, j));
            }    
          }
        }else{
          // embeddings of hashed buckets
          auto row = sp->model_->LHSEmbeddings_->row(i);
          for (unsigned int j = 0; j < args->dim; j++) {
            row(j) = (float)(embeddings(i, j));
          }    
        }
      }
      auto row = sp->model_->LHSEmbeddings_->row(i);
    }
    // This only works in case of shared embeddings
    if(!(sp->args_->shareEmb)){
      Rcpp::stop("loading from R with shareEmb to FALSE has not been implemented");
    }else{
      sp->model_->RHSEmbeddings_ = sp->model_->LHSEmbeddings_;  
    }
    if (args->isTrain) {
      // this is only needed for the backward, not the forward so only in case of training
      if(sp->args_->adagrad){
        sp->model_->LHSUpdates_.resize(sp->model_->LHSEmbeddings_->numRows());
        sp->model_->RHSUpdates_.resize(sp->model_->RHSEmbeddings_->numRows());
      }
    }
    sp->initParser();
    if(args->isTrain){
      sp->parser_->resetDict(sp->dict_);
      sp->initDataHandler();
      Rcpp::List iter = textspace_train(sp);
      if(save){
        sp->saveModel(args->model);    
      }
      out = Rcpp::List::create(
        Rcpp::Named("model") = sp,
        Rcpp::Named("args") = textspace_args(sp),
        Rcpp::Named("iter") = iter);
    }else{
      out = Rcpp::List::create(
        Rcpp::Named("model") = sp,
        Rcpp::Named("args") = textspace_args(sp)); 
    }
  }else{
    if(args->isTrain){
      if(std::ifstream(args->initModel)){
        sp->initFromSavedModel(args->initModel);
      }else{
        sp->init();  
      }
      //Rcpp::List iter = sp->train();
      Rcpp::List iter = textspace_train(sp);
      if(save){
        sp->saveModel(args->model);    
      }
      out = Rcpp::List::create(
        Rcpp::Named("model") = sp,
        Rcpp::Named("args") = textspace_args(sp),
        Rcpp::Named("iter") = iter);
    }else{
      sp->initFromSavedModel(args->model);
      sp->initDataHandler();
      sp->evaluate();
      out = Rcpp::List::create(
        Rcpp::Named("model") = sp,
        Rcpp::Named("args") = textspace_args(sp),
        Rcpp::Named("test") = "UNDER CONSTRUCTION: capture results of sp->evaluate() or write own sp->evaluate");
    } 
  }
  /*
   * Return pointer to the model and the used arguments as a list
   */
  return out;
}

// [[Rcpp::export]]
Rcpp::List textspace_evaluate(SEXP textspacemodel, std::string testFile = "", std::string basedoc = "", std::string predictionFile = "", int K = 5) {
  Rcpp::XPtr<starspace::StarSpace> sp(textspacemodel);
  sp->args_->isTrain = false; 
  sp->args_->K = K;
  if(std::ifstream(testFile)){
    sp->args_->testFile = testFile;
    sp->initDataHandler();
  }       
  if(std::ifstream(basedoc))        sp->args_->basedoc = basedoc;
  if(std::ifstream(predictionFile)) sp->args_->predictionFile = predictionFile;
  sp->evaluate();
  Rcpp::List out;
  out = Rcpp::List::create(
    Rcpp::Named("model") = sp,
    Rcpp::Named("args") = textspace_args(sp),
    Rcpp::Named("test") = "UNDER CONSTRUCTION: capture results of sp->evaluate() or write own sp->evaluate");
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
std::string textspace_save_model(SEXP textspacemodel, std::string file, bool as_tsv = false) {
  Rcpp::XPtr<starspace::StarSpace> sp(textspacemodel);
  if(as_tsv){
    sp->saveModelTsv(file);  
  }else{
    sp->saveModel(file);   
  }
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
    for(unsigned int j = 0; j < vec.numCols(); j++){
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
    for(unsigned int j = 0; j < vec.size(); j++){
      embedding(i, j) = vec[j];
    }
  }
  return embedding;
}


// [[Rcpp::export]]
Rcpp::NumericMatrix textspace_embedding_lhsrhs(SEXP textspacemodel, std::string type = "lhs") {
  Rcpp::XPtr<starspace::StarSpace> sp(textspacemodel);

  if(type == "lhs"){
    Rcpp::NumericMatrix embedding(sp->model_->LHSEmbeddings_->numRows(), sp->args_->dim);
    for (unsigned int i = 0; i < sp->model_->LHSEmbeddings_->numRows(); i++){
      auto row = sp->model_->LHSEmbeddings_->row(i);
      for (unsigned int j = 0; j < sp->args_->dim; j++) {
        embedding(i, j) = row(j);
      }    
    }
    return embedding;
  }else if(type == "rhs"){
    Rcpp::NumericMatrix embedding(sp->model_->RHSEmbeddings_->numRows(), sp->args_->dim);
    for (unsigned int i = 0; i < sp->model_->RHSEmbeddings_->numRows(); i++){
      auto row = sp->model_->RHSEmbeddings_->row(i);
      for (unsigned int j = 0; j < sp->args_->dim; j++) {
        embedding(i, j) = row(j);
      }    
    }
    return embedding;
  }else{
    Rcpp::stop("type should be either lhs or rhs");
  }
}

// [[Rcpp::export]]
Rcpp::List textspace_predict(SEXP textspacemodel, std::string input, int k = 5, Rcpp::StringVector basedoc = "", std::string sep = " ") { 
  Rcpp::XPtr<starspace::StarSpace> sp(textspacemodel);
  //Rcpp::Rcout << "ngrams: " << sp->args_->ngrams << "\n";
  // Set number of elements to predict
  sp->args_->K = k;
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
  Rcout << sp->baseDocs_.size() << endl;
  for(int i=0; i < sp->baseDocs_.size(); i++){
    Rcout << sp->dict_->getSymbol(sp->baseDocs_[i][0].first) << endl;
  }
  */
  //Rcpp::Rcout << "basedocs: " << sp->baseDocs_.size() << "\n";

  // Do the prediction
  std::vector<starspace::Base> query_vec;
  std::vector<starspace::Predictions> predictions;
  std::vector<starspace::Base> tokens;
  // split according to separator of the input query and put in query_vec all tokens which are part of the dictionary only
  sp->parseDoc(input, query_vec, sep);
  //Rcpp::Rcout << "query_vec size: " << query_vec.size() << "\n";
  sp->predictOne(query_vec, predictions);
  //Rcpp::Rcout << "predictions size: " << predictions.size() << "\n";
  std::vector<std::string> label;
  std::vector<std::string> basedoc_terms;
  std::vector<int32_t> basedoc_index;
  std::vector<float> prob;
  for (unsigned int i = 0; i < predictions.size(); i++) {
    prob.push_back(predictions[i].first);
    basedoc_index.push_back(predictions[i].second + 1); 
    tokens = sp->baseDocs_[predictions[i].second]; 
    //Rcpp::Rcout << "tokens size: " << tokens.size() << "\n";
    //Rcpp::Rcout << "predictions[i].first: " << predictions[i].first << "\n";
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
          //Rcpp::Rcout << "getSymbol(t.first): " << sp->dict_->getSymbol(t.first) << "\n";
          label.push_back(sp->dict_->getSymbol(t.first));
        }
      }
    }
  }
  Rcpp::List out = Rcpp::List::create(Rcpp::Named("input") = input, 
                                      Rcpp::Named("prediction") = Rcpp::DataFrame::create(
                                        Rcpp::Named("label_starspace") = label,
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
      Rcout << vec.cell(i,j) << ' ';    
    }  
   Rcout << endl;    
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


