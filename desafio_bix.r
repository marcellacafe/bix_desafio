
# Instalação da biblioteca DMwR que não está mais disponível no CRAN. Para instalá-la é preciso ter o pacote devtools instalado.
# devtools::install_github("cran/DMwR")

options(scipen = 999) # Retira a notacao cientifica

# Carregando as bibliotecas utilizadas
library(caret)
library(corrplot)
library(DMwR)
library(e1071)
library(factoextra)
library(FactoMineR)
library(mice)
library(naniar)
library(plotly)
library(qcc)
library(randomForest)
library(reactable)
library(readxl)
library(ROSE)
library(tidyverse)
library(VIM)
library(xgboost)

# Carregando a base de dados de treino (anos anteriores) e teste (2020)
df_train <- read.csv("base_vigente_anos_anteriores.csv", header = T, na.strings = "na")
df_test <- read.csv("base_vigente_2020.csv", header = T, na.strings = "na")

# Checando a dimensão da base de dados de treino
dim(df_train)

# Verificando os primeiros dados da base de dados de treino
head(df_train)

# Verificando o nome das colunas do dataset de treino
dput(colnames(df_train))


# Verificando o resumo da base de dados de treino
summary(df_train)

# Quantidade de dados da classe com valor "pos"
qtd_pos <- df_train %>%
  dplyr::filter(class == "pos")

dim(qtd_pos)
# Quantidade de dados da classe com valor "neg"
qtd_neg <- df_train %>%
  dplyr::filter(class == "neg")

dim(qtd_neg)

# Dimensão do dataset de teste
dim(df_test)

# Distribuição das classes do dataset de treino
plotly::plot_ly(data = df_train, x = ~class, type = "histogram")

# Distribuição das classes do dataset de teste
plotly::plot_ly(data = df_test, x = ~class, type = "histogram")

# Verificando variáveis que possuem mais dados ausentes
naniar::gg_miss_var(df_train)

var_na <- stack(100 * colSums(is.na(df_train)) / nrow(df_train))
order_var_na <- arrange(var_na, desc(values))

order_var_na$ind <- factor(order_var_na$ind, levels = unique(order_var_na$ind)[order(order_var_na$value, decreasing = TRUE)])

# Gráfico das top 15 variáveis que possuem mais valores NA
plotly::plot_ly(order_var_na[c(1:15), ], x = ~ind, y = ~values, type = "bar") %>%
  layout(title = "Top 15 valores NA")

# Removendo colunas com mais de 70% de dados ausentes
dt_train <- df_train[, -which(colMeans(is.na(df_train)) > 0.7)]
dt_test <- df_test[, -which(colMeans(is.na(df_test)) > 0.7)]

# A coluna "cd_000" é constante, portanto, será eliminada do dataset
dt_train <- subset(dt_train, select = -c(cd_000))
dt_test <- subset(dt_test, select = -c(cd_000))

# A imputação que será realizada para os outros valores nulos será preenchendo os valores nulos com a mediana
train_imp <- dt_train[, -c(1, 2)]

for (i in 1:ncol(train_imp)) {
  train_imp[is.na(train_imp[, i]), i] <- median(train_imp[, i], na.rm = TRUE)
}

df_train_imp <- cbind(dt_train[, c(1, 2)], train_imp)

# O mesmo será feito com o dataset de teste

test_imp <- dt_test[, -c(1, 2)]

for (i in 1:ncol(test_imp)) {
  test_imp[is.na(test_imp[, i]), i] <- median(test_imp[, i], na.rm = TRUE)
}

df_test_imp <- cbind(dt_test[, c(1, 2)], test_imp)

# Checando se após imputar os valores ausentes com a mediana há valores ausentes no conjunto de dados.
naniar::gg_miss_var(df_train_imp)
naniar::gg_miss_var(df_test_imp)

cor_matrix <- cor(x = df_train_imp[, -1], method = "spearman", use = "complete.obs")

corrplot::corrplot(cor(df_train_imp[, -1]), type = "lower", method = "ellipse", title = "Correlação entre as variáveis")

# Buscando as colunas que devem ser removidas para reduzir a correlação entre pares
high_cor <- caret::findCorrelation(cor_matrix, cutoff = 0.75, names = TRUE)
high_cor <- dput(high_cor)

# Quantidade de variáveis
length(high_cor)

# Eliminar variáveis com correlação alta
df_train_X <- df_train_imp %>%
  dplyr::select(!high_cor)

df_test_X <- df_test_imp %>%
  dplyr::select(!high_cor)

pca <- FactoMineR::PCA(df_train_X[, -1], graph = TRUE)
autovalores <- factoextra::get_eigenvalue(pca)
factoextra::fviz_eig(pca, addlabels = TRUE, ylim = c(0, 50))
data.table(autovalores)

# PCA no dataset de treino

# Análise de Componentes Principais
pri <- stats::prcomp(df_train_X[, -1], center = TRUE, scale = TRUE)
df_pca <- as.data.frame(pri$x)
df_pca <- cbind(df_pca, df_train_imp[1])
df_pca_best <- df_pca[, c(1:26, 71)]
head(df_pca_best)

# PCA no dataset de test
df_pca_test_best <- predict(pri, df_test_imp[-1])
df_pca_test <- cbind(df_pca_test_best, df_test_imp[1]) %>% as.data.frame()
df_pca_test <- df_pca_test[, c(1:26, 71)]
head(df_pca_test)

# Over Sampling

df_pca_best$class <- ifelse(df_pca_best$class == "neg", 0, 1)

df_over <- ROSE::ovun.sample(class ~ .,
                            data = df_pca_best,
                            p = 0.7, seed = 1,
                            method = "over"
                            )$data

table(df_over$class)
logit_over <- stats::glm(class ~ ., data = df_over, family = "binomial")
logit_over_pred <- stats::predict(logit_over, df_over, type = "response")
over_pred <- as.data.frame(ifelse(logit_over_pred > 0.5, 1, 0))
names(over_pred) <- c("class")
confusionMatrix(factor(over_pred$class), factor(df_over$class))

# Under Sampling

data.balanced.under <- ovun.sample(class~., data=df_pca_best, 
                                   p=0.5, seed=1, 
                                   method="under")$data
table(data.balanced.under$class)
logit.under <- glm(class~., data = data.balanced.under, family = "binomial") 
logit.under.pred <- predict(logit.under, data.balanced.under, type = "response")
under.pred <- as.data.frame(ifelse(logit.under.pred > 0.5, 1, 0))
names(under.pred) = c("class")
confusionMatrix(factor(under.pred$class),factor(data.balanced.under$class))

# SMOTE 

# Transformando a variável de destino no tipo factor
df_class_factor <- df_pca_best
df_class_factor$class <- as.factor(df_class_factor$class)

data.smote <- DMwR::SMOTE(class~., df_class_factor, perc.over = 1900, perc.under = 210.53, k=5)

table(data.smote$class)
logit.smote <- glm(class~., data = data.smote, family = "binomial") 
logit.smote.pred <- predict(logit.smote, data.smote, type = "response")
smote.pred <- as.data.frame(ifelse(logit.smote.pred > 0.5, 1, 0))
names(smote.pred) = c("class")
confusionMatrix(factor(smote.pred$class),factor(data.smote$class))

# Under e Over Sampling

data.both = ovun.sample(class~., data=df_pca_best, p=0.5, seed=1, 
                        method="both")$data
table(data.both$class)
logit.both <- glm(class~., data = data.both, family = "binomial") 
logit.both.pred <- predict(logit.both, data.both, type = "response")
both.pred <- as.data.frame(ifelse(logit.both.pred > 0.5, 1, 0))
names(both.pred) = c("class")
confusionMatrix(factor(both.pred$class),factor(data.both$class))

pca.best.smote <- DMwR::SMOTE(class~., df_class_factor, perc.over = 1900, perc.under = 210.53, k=5)
table(pca.best.smote$class)

# SVM Linear

df_pca_test_class <- df_pca_test
df_pca_test_class$class <- ifelse(df_pca_test_class$class == "neg", 0, 1)
df_pca_test_class$class <- as.factor(df_pca_test_class$class)

svm.lin <- e1071::svm(class~., data=pca.best.smote, kernel='linear', cost=0.01)
summary(svm.lin)
pred_lin <- stats::predict(svm.lin, df_pca_test)
cm <- confusionMatrix(pred_lin, df_pca_test_class$class)

custo_total <- function(FP, FN){
  custo <- (FP * 10) + (FN * 500)
  return(custo)
}

# Custo utilizando o algoritmo SVM Linear
custo <- custo_total(cm$table[2,1], cm$table[1,2])
paste0("Custo do SVM Linear é: $", custo)

# SVM com Kernel Radial

svm_rad <- e1071::svm(class~., data=pca.best.smote, kernel='radial', gamma=0.1, cost=0.1)
summary(svm_rad)
pred_rad <- stats::predict(svm_rad, df_pca_test)
cm_svm_rad <- confusionMatrix(pred_rad, df_pca_test_class$class)
cm_svm_rad

# Custo utilizando o algoritmo SVM Radial
custo <- custo_total(cm_svm_rad$table[2,1], cm_svm_rad$table[1,2])
paste0("Custo do SVM Radial é: $", custo)

# SVM com Kernel Polinomial

svm_poly <- svm(class~., data=pca.best.smote, kernel='polynomial', cost=0.01,gamma=0.1, degree=2)
summary(svm_poly)
pred_ploy <- predict(svm_poly, df_pca_test)
cm_svm_poly <- confusionMatrix(pred_ploy, df_pca_test_class$class)
cm_svm_poly

# Custo utilizando o algoritmo SVM Polinomial
custo <- custo_total(cm_svm_poly$table[2,1], cm_svm_poly$table[1,2])
paste0("Custo do SVM Radial é: $", custo)

# Random Forest

set.seed(1111)
rf <- randomForest::randomForest(class~., data = pca.best.smote, ntree = 500, mtry = 6, importance= TRUE)
pred_rf <- stats::predict(rf, df_pca_test, type = "class")
cm_rf <- confusionMatrix(pred_rf, df_pca_test_class$class)
cm_rf

# Custo utilizando o algoritmo Random Forest
custo <- custo_total(cm_rf$table[2,1], cm_rf$table[1,2])
paste0("Custo do Random Forest é: $", custo)

# XGBoost 

train_label = pca.best.smote[,'class']
train_label = as.integer(train_label) -1
train_matrix = xgboost::xgb.DMatrix(data = as.matrix(pca.best.smote[-27]), label=train_label)

test_label = df_pca_test[,'class']
test_label = as.integer(test_label) - 1
test_matrix = xgboost::xgb.DMatrix(data = as.matrix(df_pca_test[-27]), label=test_label)

set.seed(1111)
xgb <- xgboost(data = train_matrix,
               eta = 0.4,
               max_depth = 6, 
               nround=100, 
               subsample = 0.5,
               min_child_weight = 2,
               colsample_bytree = 0.5,
               seed = 1111, 
               gamma = 100,
               eval_metric = "error",
               objective = "binary:logistic",
               nthread = 3)

y_pred <- round(predict(xgb, data.matrix(df_pca_test[,-27])))

y_pred[y_pred == 0] <- c("neg")
y_pred[y_pred == 1] <- c("pos")
cm_xgb <- confusionMatrix(factor(y_pred), factor(df_pca_test$class))
cm_xgb

# Custo utilizando o algoritmo XGBoost
custo <- custo_total(cm_xgb$table[2,1], cm_xgb$table[1,2])
paste0("Custo do XGBoost é: $", custo)
