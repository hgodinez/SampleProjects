library(h2o)
library(readr)
library(tidyverse)
library(ggplot2)


#The task at hand is a binary classification problem, for which both a training and a validation set are
# provided as csv files. 
# Task to building a classifier.

# Read the data. I'm using read_csv2 from the "readr" package since the numeric values are separated by "," 
# from the decimals. Other ways to do it are using regular expresion
# and data.table (data.table package is much faster  for reading and writing)

cfd_ <- read_csv2("data/Training.csv",
                             locale = locale("de"),
                             col_names = TRUE
                             
)

cfd_ <- cfd_ %>% mutate_if(is.character,as.factor)

# I did a under-sampling to get a more balanced training set
cfd_.no<-cfd_[cfd_$classlabel=='no.',]
cfd_.yes<-cfd_[cfd_$classlabel=='yes.',]
cfd_.yes<-cfd_.yes[sample(0.1*nrow(cfd_.yes)),]
cfd_<-rbind(cfd_.no,cfd_.yes)

h2o.init(max_mem_size = "16g")

#Split data into Train/Validation
cfd_<-as.h2o(cfd_)
split_h2o <- h2o.splitFrame(cfd_, c(0.8), seed = 1234 )
train_credit_v2 <- h2o.assign(split_h2o[[1]], "train" ) # 80%
valid_credit_v2 <- h2o.assign(split_h2o[[2]], "valid" ) # 20% to evaluate  the models without using the validation test

# let's import the validation data

test_validation_credit_v2 <- read_csv2("data/Validation.csv",
                           locale = locale("de"),
                           col_names = TRUE
                           
)

# I'm converting all character variables to factors for using in H2O 
test_validation_credit_v2 <- test_validation_credit_v2 %>% mutate_if(is.character,as.factor)

test_validation_credit_v2<-as.h2o(test_validation_credit_v2) # this validation set is used only for prediction. 

#Model
# Set names for h2o
target <- "classlabel"
predictors <- setdiff(names(train_credit_v2), target)
# if you want to run the model using all the features (including v68) comment the next line.
predictors<-setdiff(predictors, "v68") 
####



#############################################################################################################################################
#####              FIRST MODEL: GBM tuned using grid search                                                                              ####
#############################################################################################################################################


# I'll train a GBM then I'll tune the parametrs
gbm_credit <- h2o.gbm(
  x = predictors, 
  y = target, 
  training_frame = train_credit_v2, 
  validation_frame = valid_credit_v2,
  ntrees = 10000,                                                            
  learn_rate=0.0001,                                                         
  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC", 
  sample_rate = 0.8,                                                       
  col_sample_rate = 0.8,
  seed = 159, 
  score_tree_interval = 10,
  nfolds=10
)

## Hyper parameter search for max and min depth
hyper_params = list( max_depth = seq(1,40,2) )

grid <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = list(strategy = "Cartesian"),
  algorithm="gbm",
  grid_id="depth_grid",
  x = predictors, 
  y = target, 
  training_frame = train_credit_v2, 
  validation_frame = valid_credit_v2,
  ntrees = 100000,
  learn_rate = 0.03, 
  learn_rate_annealing = 0.99,
  sample_rate = 0.8, 
  col_sample_rate = 0.8,
  seed = 1234,
  stopping_rounds = 5,
  stopping_tolerance = 1e-4,
  stopping_metric = "AUC", 
  score_tree_interval = 10
)

                                                                 

## sort the grid models by decreasing AUC
sortedGrid <- h2o.getGrid("depth_grid", sort_by="AUC", decreasing = FALSE)    
#sortedGrid

## find the range of max_depth for the top 5 models
topDepths = sortedGrid@summary_table$max_depth[1:5]                       
minDepth = min(as.numeric(topDepths))
maxDepth = max(as.numeric(topDepths))
minDepth
maxDepth

# we do a random search for optimizing other parameters
## Random grid search
hyper_params = list( 
  max_depth = seq(minDepth,maxDepth,1),  
  sample_rate = seq(0.1,1,0.01),  
  col_sample_rate = seq(0.1,1,0.01), 
  col_sample_rate_per_tree = seq(0.1,1,0.01),  
  col_sample_rate_change_per_level = seq(0.9,1.1,0.01),  
  min_rows = 2^seq(0,log2(nrow(train_credit_v2))-1,1), 
  nbins = 2^seq(4,10,1),              
  nbins_cats = 2^seq(4,12,1),   
  min_split_improvement = c(0,1e-8,1e-6,1e-4), 
  histogram_type = c("UniformAdaptive","QuantilesGlobal","RoundRobin")       
)

search_criteria = list(
  strategy = "RandomDiscrete",      
  max_runtime_secs = 300,   #could be slow (setting to just 5 mins)      
  max_models = 200,               
  seed = 159,  
  stopping_rounds = 5,                
  stopping_metric = "AUC",
  stopping_tolerance = 1e-3
)




grid <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "gbm",
  grid_id = "final_grid",
  x = predictors, 
  y = target, 
  training_frame = train_credit_v2, 
  validation_frame = valid_credit_v2,
  ntrees = 10000,                                         
  learn_rate = 0.03,                   
  learn_rate_annealing = 0.99,                                               
  max_runtime_secs = 3600,                                                 
  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC", 
  score_tree_interval = 10,                                                
  seed = 1234
)

## Sort the grid models by AUC
sortedGrid <- h2o.getGrid("final_grid", sort_by = "AUC", decreasing = FALSE)    
#sortedGrid



gbm_credit_v2_v68 <- h2o.getModel(sortedGrid@model_ids[[1]])
print(h2o.auc(h2o.performance(gbm_credit_v2_v68, newdata = test_validation_credit_v2)))

#gbm_credit_v2_v68@parameters
perf_credit_v2 <- h2o.performance(gbm_credit_v2_v68,test_validation_credit_v2)
h2o.confusionMatrix(perf_credit_v2)


#let's use the full dataset with the tuned model 
model_final_credit_v2_v68 <- do.call(h2o.gbm,
                 ## update parameters in place
                 {
                   p <- gbm_credit_v2_v68@parameters
                   p$model_id = NULL          
                   p$training_frame = cfd_     
                   p$validation_frame = NULL  ## no validation frame
                   p$nfolds = 10               ## cross-validation
                   p
                   
                 }
)
#model_final_credit_v2_v68@model$cross_validation_metrics_summary

perf_credit_v2_full <- h2o.performance(model_final_credit_v2_v68,test_validation_credit_v2)
h2o.confusionMatrix(perf_credit_v2_full)



pred_conversion <- as.data.frame(h2o.predict(object = model_final_credit_v2_v68 , newdata = test_validation_credit_v2)) %>%
  mutate(actual = as.vector(test_validation_credit_v2[, c('classlabel')]))


pred_conversion %>%
  mutate(predict = ifelse(pred_conversion$no. >(1-h2o.find_threshold_by_max_metric(perf_credit_v2_full, 'f1')[1]), "no.", "yes.")) %>%
  group_by(actual, predict) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n)) 




#############################################################################################################################################
#####              SECOND MODEL: Deep Learning AutoEncoder as a pretrained model for a supervised Deep Learning model                    ####
#############################################################################################################################################


### We train an unsupervised model (autoencoder)
dp_credit_v2 <- h2o.deeplearning(x = predictors, 
                             training_frame = cfd_,
                             model_id = "dp_credit_v2",
                             autoencoder = TRUE,
                             ignore_const_cols = FALSE,
                             seed = 345,
                             hidden = c(6, 2, 6,6), 
                             epochs = 100,
                             activation = "Tanh")



# let's take the 3rd hidden layer as dimensionality reduction
train_features <- h2o.deepfeatures(dp_credit_v2, cfd_, layer = 3) %>%
  as.data.frame() %>%
  mutate(classlabel = as.factor(as.vector(cfd_[, c('classlabel')]))) %>%
  as.h2o()

#extract the classlabel from the features 
features_dim <- setdiff(colnames(train_features), target)

#let's train a NN using the features extracted (dimensionality reduction)
dp_credit_v2_dim <- h2o.deeplearning(y = target,
                                 x = features_dim,
                                 training_frame = train_features,
                                 ignore_const_cols = FALSE,
                                 seed = 42,
                                 hidden = c(6, 2,6,6), 
                                 epochs = 100,
                                 activation = "Tanh",
                                 nfolds=10)
#dp_credit_v2_dim

#evaluate in the validation set
test_dim <- h2o.deepfeatures(dp_credit_v2, test_validation_credit_v2, layer = 3)

h2o.predict(dp_credit_v2_dim, test_dim) %>%
  as.data.frame() %>%
  mutate(actual = as.vector(test_validation_credit_v2[, c('classlabel')])) %>%
  group_by(actual, predict) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n))


#now lets train another NN but this time supervised using the pre-training autoencoder model

dp_credit_v2_2 <- h2o.deeplearning(y = target,
                               x = predictors,
                               training_frame = cfd_,
                               pretrained_autoencoder  = "dp_credit_v2",
                                ignore_const_cols = FALSE,
                               seed = 42,
                               hidden = c(6, 2, 6,6), 
                               epochs = 100,
                               activation = "Tanh")

#dp_credit_v2_2

#predicting in the validation set. 
pred <- as.data.frame(h2o.predict(object = dp_credit_v2_2, newdata = test_validation_credit_v2)) %>%
  mutate(actual = as.vector(test_validation_credit_v2[, c('classlabel')]))


pred %>%
  group_by(actual, predict) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n)) 




