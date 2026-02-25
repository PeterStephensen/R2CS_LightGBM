library(lightgbm)

set.seed(42)
n <- 1000

x1 <- rnorm(n, mean = 35, sd = 10)
x2 <- rnorm(n, mean = 50000, sd = 15000)
x3 <- runif(n, 0, 100)
x4 <- rpois(n, lambda = 5)

log_odds <- -2 + 0.03 * x1 + 0.00002 * x2 + 0.02 * x3 - 0.1 * x4
prob <- 1 / (1 + exp(-log_odds))
y <- rbinom(n, 1, prob)

features <- data.frame(x1 = x1, x2 = x2, x3 = x3, x4 = x4)

train_idx <- sample(1:n, 800)
train_features <- as.matrix(features[train_idx, ])
train_label <- y[train_idx]
test_features <- as.matrix(features[-train_idx, ])
test_label <- y[-train_idx]

dtrain <- lgb.Dataset(train_features, label = train_label)

params <- list(
  objective   = "binary",
  metric      = "binary_logloss",
  num_leaves  = 31,
  learning_rate = 0.05,
  feature_fraction = 0.9,
  bagging_fraction = 0.8,
  bagging_freq = 5,
  verbose = -1
)

model <- lgb.train(
  params  = params,
  data    = dtrain,
  nrounds = 100,
  verbose = -1
)

test_pred <- predict(model, test_features)

dir.create("model", showWarnings = FALSE)
dir.create("data",  showWarnings = FALSE)

lgb.save(model, "model/lightgbm_model.txt")

write.csv(
  data.frame(x1 = test_features[, 1],
             x2 = test_features[, 2],
             x3 = test_features[, 3],
             x4 = test_features[, 4]),
  "data/test_features.csv",
  row.names = FALSE
)

write.csv(
  data.frame(probability = test_pred),
  "data/test_predictions_r.csv",
  row.names = FALSE
)

cat(sprintf("Træningssamples: %d\n", length(train_label)))
cat(sprintf("Testsamples:     %d\n", length(test_label)))
cat(sprintf("Positiv rate (test): %.2f%%\n", 100 * mean(test_label)))
cat(sprintf("AUC (approx):    %.4f\n",
    {
      ord <- order(test_pred)
      ranks <- rank(test_pred)
      n1 <- sum(test_label == 1)
      n0 <- sum(test_label == 0)
      (sum(ranks[test_label == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
    }))
cat("\nModel gemt:       model/lightgbm_model.txt\n")
cat("Testdata gemt:    data/test_features.csv\n")
cat("R-predictions:    data/test_predictions_r.csv\n")
