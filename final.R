# Perform all steps for my capstone project submission.
#
# This script includes the following steps.
#
# 0. Install and load required packages.
# 1. Process raw data to extract features and outcomes.
# 2. Conduct exploratory data analysis.
# 3. Train and evaluate machine learning models.


# Step 0: Install and load packages. ==========================================
if (!require(caret)) install.packages("caret")
if (!require(e1071)) install.packages("e1071")
if (!require(GGally)) install.packages("GGally")
if (!require(ggpubr)) install.packages("ggpubr")
if (!require(lubridate)) install.packages("lubridate")
if (!require(tidyverse)) install.packages("tidyverse")
library(caret)
library(GGally)
library(ggpubr)
library(lubridate)
library(rstatix)
library(tidyverse)


# Step 1: Process and partition raw data. =====================================

# Read processed data from disk if available. Otherwise create the processed
# data from raw. Note that processing raw data could take several minutes.
if (file.exists("processed.csv")) {
  processed <-
    read_csv("processed.csv", 
      col_types = cols(is_repeat = col_factor())
    ) %>% 
    select(-donor_id)
} else {

  # Read donor and donation data.
  donations <- 
    read_csv("Donations.csv") %>% 
    rename_with(~ tolower(gsub(" ", "_", .x)))

  # Establish a cutoff date for past and future donations.
  cutoff <- ymd("2017-05-01")

  # Identify donors who donate at or after the cutoff.
  future_donors <- 
    donations %>% 
    filter(donation_received_date >= cutoff) %>% 
    select(donor_id) %>% 
    unique() %>% 
    pull()
  
  # Calculate the seconds between a start and end time.
  seconds_since <- function(start, end){
    int_length(interval(start, end))
  }
  
  # Process the source data for the ML task.
  # 
  # 1. Filter to donations that occurred before the cutoff.
  # 2. Summarize donation characteristics by donor (features).
  # 3. Identify donors that donated again (outcome).
  processed <- 
    donations %>% 
    filter(donation_received_date < cutoff) %>% 
    group_by(donor_id) %>% 
    summarise(
      n_donations = n(),
      seconds_ago = seconds_since(max(donation_received_date), cutoff),
      years_ago = seconds_ago / 60 / 60 / 24 / 365,
      mean_amount = mean(donation_amount)
    ) %>% 
    select(-seconds_ago) %>% 
    mutate(is_repeat = as.factor(donor_id %in% future_donors))
  
  # Cache the processed data for quick reading next time.
  write_csv(processed, "processed.csv")
}

# Partition the train and test data. Note that for reasonable processing times,
# you can sample cases from the training set rather than using all of them.
set.seed(0)
index <- createDataPartition(processed$is_repeat, p = .9, list = FALSE)
trn <- processed[c(index),] %>% sample_n(size = 50000)  # Adjust as needed.
tst <- processed[-c(index),]


# Step 2: Explore the data. ===================================================

# Plot a lower cell in a pairplot.
lower <- function(data, mapping, ...){
  data %>% ggplot(mapping = mapping) +
    geom_point(alpha = 0.1)
}

# Plot a diagonal cell in a pairplot.
diag <- function(data, mapping, ...){
  data %>% ggplot(mapping = mapping) +
    geom_density(alpha = 0.5)
}

# Fix a subplot by setting given axes to log scale.
fix_plot <- function(plt, x = FALSE, y = FALSE){
  fixed <- plt
  if (x) {fixed <- fixed + scale_x_log10()}
  if (y) {fixed <- fixed + scale_y_log10()}
  fixed$type <- "logcontinuous"
  fixed$subType <- "logpoints"
  fixed
}

# Create a pair plot of all features.
plt_pairs <- 
  trn %>% 
  ggpairs(
    columns = c("n_donations", "years_ago", "mean_amount"), 
    title = "Relationships Between Features and Outcomes",
    columnLabels = c("Number of Donations",
                     "Years Since Donation",
                     "Average Amount"),
    mapping = aes(fill = is_repeat, color = is_repeat),
    legend = 7,
    upper = list(continuous="blank"),
    lower = list(continuous=lower),
    diag = list(continuous=diag)
  )

# Adjust individual plots in the pairplot.
plt_pairs[1, 1] <- 
  fix_plot(plt_pairs[1, 1], x = TRUE) +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
plt_pairs[2, 1] <- fix_plot(plt_pairs[2, 1], x = TRUE)
plt_pairs[3, 1] <- fix_plot(plt_pairs[3, 1], x = TRUE, y = TRUE)
plt_pairs[3, 2] <- fix_plot(plt_pairs[3, 2], y = TRUE)
plt_pairs[3, 3] <- fix_plot(plt_pairs[3, 3], x = TRUE)

# View and save the pairplot.
plt_pairs
ggsave("explore.png", plt_pairs)


# Step 3: Train and evaluate models. ==========================================

# 3.1: Set up for training models. --------------------------------------------

# Specify a 5-fold cross validation approach for models.
control <- trainControl(
  method = "cv",
  number = 5
)

# This project uses a simple yar cutoff heuristic as a baseline. For a given
# year cutoff, the heuristic classifies all cases under the cutoff as repeat
# donors (TRUE) and everyone else as non-repeat donors (FALSE).
#
# Rather than eyeballing a good cutoff, I use caret to tune the cutoff. The
# rest of the code in this section gives caret the information it needs to use
# year cutoffs in its parameter tuning infrastructure.

# First, specify a model for a simple year cutoff heuristic. This heuristic
# doesn't actually "fit" any data; the single cutoff parameter predicts classes
# directly. This function just helps carat find the parameter for making
# predictions (e.g., during parameter tuning).
year_cutoff_fit <- function(param, ...) {
  list(cutoff = param)
}

# Next, specify a function to predict classes basd on the simple year cutoff.
# This function predicts TRUE for values below the cutoff and FALSE otherwise.
year_cutoff_predict <- function(modelFit, newdata, ...) {
  as.factor(newdata < pull(modelFit$cutoff))
}

# Finally, combine fitting and predicting functions with other metadata so that
# caret can tune the best year cutoff heuristic.
year_cutoff_method <- list(
  library = NULL,
  type = "Classification",
  parameters = data.frame(
    parameter = "cutoff",
    class = "numeric",
    label = "Cutoff"
  ),
  grid = function(...) data.frame(cutoff=seq(0, 4, 0.25)),
  fit = year_cutoff_fit,
  predict = year_cutoff_predict,
  prob = NULL
)

# 3.2: Train the candidate models. --------------------------------------------

# Train using a simple year cutoff heuristic.
set.seed(0)
fit0 <- train(
  is_repeat ~ years_ago,
  data = trn,
  method = year_cutoff_method,
  trControl = control,
  metric = "Kappa"
)

# Train a logistic model using number of donations and years ago.
set.seed(0)
fit1 <- train(
  is_repeat ~ n_donations + years_ago,
  trn,
  method = "glm",
  family = "binomial",
  trControl = control,
  metric = "Kappa"
)

# Train a KNN model using number of donations and years ago.
set.seed(0)
fit2 <- train(
  is_repeat ~ years_ago + n_donations,
  trn,
  method = "knn",
  metric = "Kappa",
  tuneGrid = data.frame(k = 2^(1:8)),
  trControl = control
)

# Plot and save parmeter tuning results.
plt_tune0 <- 
  ggplot(fit0, highlight = TRUE) + 
  labs(title = "Optimal Year Cutoff")

plt_tune2 <-
  ggplot(fit2, highlight = TRUE) +
  labs(title = "Optimal #Neighbors")

plt_tune0
plt_tune2

ggsave("cutoff.png", plt_tune0)
ggsave("knn.png", plt_tune2)


# 3.3: Explore decision boundaries. -------------------------------------------

# Use a grid of predictions to create KNN decision contours. The values for
# n_observations are spaced out on a log scale so that they are equidistant
# when plotted using a log scale for n_observations. Skipping this
# transformation produces large spans on the canvas with no predictions in the
# left half of the plot, resulting in distorted, angular contours.
grid <- 
  expand.grid(
    n_donations = 10^(seq(0, log(max(tst$n_donations)), length.out=200)),
    years_ago = seq(0, max(tst$years_ago), length.out = 200)
  ) %>% 
  as_tibble() %>% 
  mutate(is_repeat = unclass(predict(fit2, .)))

# Identify points along the decision boundary for the logistic model. The
# version of ggplot2 used for this project does not support geom_abline for
# plots with a log-scaled axis, so I can't just plot the line with an equation.
# As a workaround, I have ggplot2 fit a model to a few points on the line.
b0 <- fit1$finalModel$coefficients[1]  # Intercept
b1 <- fit1$finalModel$coefficients[2]  # Slope for n_donations.
b2 <- fit1$finalModel$coefficients[3]  # Slope for years_ago.
fit1_boundary <- tibble(
  years_ago = seq(
    min(tst$years_ago),
    max(tst$years_ago), length.out = 10
  ),
  # Line representing 50/50 odds.
  n_donations = -b0 / b1 + -b2 / b1 * years_ago
)

# Use a scatterplot of test cases as a base plot.
base <- 
  tst %>%
  ggplot(aes(n_donations, years_ago)) +
  geom_point(aes(color = is_repeat), alpha = 0.1) +
  scale_x_log10() +  # Use a log scale for right-skewed n_donations.
  guides(color = guide_legend(title = "Repeat Donor")) +
  theme(axis.title = element_blank())

# Plot the year cutoff heuristic boundary.
plot0 <- base + geom_hline(yintercept = fit0$bestTune$cutoff)

# Plot the logistic model boundary.
plot1 <- base + geom_line(data = fit1_boundary)

# Plot the KNN model boundaries.
plot2 <- 
  base + 
  geom_contour(
    aes(z = is_repeat),
    data = grid,
    breaks = 1.5,
    color = "black"
  )

# Arrange boundary plots into an annotated figure.
fig <- 
  ggarrange(
    plot0, plot1, plot2,
    nrow = 1, 
    labels = c("Year Cutoff", "Logistic", "KNN"),
    legend = "right",
    common.legend = TRUE
  ) %>% 
  annotate_figure(
    top = text_grob("Decision Boundaries by Model"),
    left = "Years Ago",
    bottom = "Donations to Date"
  )

# View and save the figure.
fig
ggsave("model.png", fig)

# 3.4: Evaluate candidate model performance. ----------------------------------

# Extract a list of results from a confusion matrix.
assemble_results <- function(cm, name){
  model <- c(Model=str_to_title(name))
  overall <- cm[["overall"]][c("Kappa", "Accuracy")]
  byclass <- cm[["byClass"]][c("Sensitivity", "Specificity")]
  flatten(list(model, overall, byclass))
}

# Evaluate and report the final results.
results <- 
  list(cutoff=fit0, logistic=fit1, knn=fit2) %>% 
  map(~ predict(.x, tst)) %>% 
  map(confusionMatrix, tst$is_repeat, positive = "TRUE") %>% 
  imap_dfr(assemble_results) %>%
  mutate_if(is.numeric, ~ round(., digits = 3))

# View results.
results
