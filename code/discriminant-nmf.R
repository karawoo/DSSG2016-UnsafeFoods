##################################################
####  Discriminant NMF on Amazon review data  ####
##################################################

## Load required packages
library("jsonlite")
library("DNMF")
library("tm")
library("dplyr")
library("tidyr")
library("ggplot2")

## Load Amazon review data
json_file <- "../data/raw/reviews_Grocery_and_Gourmet_Food.json"
amz <- stream_in(file(json_file))

## Load list of recalled products
recalled <- read.csv("../data/processed/recalls_upcs_asins_joined.csv",
                     stringsAsFactors = FALSE)

## Extract vector of ASINs of recalled products
recalled_asins <- unique(recalled$asins) %>%
  sapply(strsplit, ";") %>%
  unname() %>%
  unlist()

## Add recalled/not recalled column to Amazon reviews based on vector of ASINs
amz <- mutate(amz, recalled = ifelse(asin %in% recalled_asins, 1, 2))

## Set seed for reproducibility
set.seed(123)

## Subset data -- 1000 rows each for recalled and non-recalled reviews
amz_sub <- amz %>%
  group_by(recalled) %>%
  sample_n(1000)
  
## Vector of Amazon review text
reviews <- amz_sub$reviewText

## Create corpus
corp <- Corpus(VectorSource(reviews))

## Clean up data
corp <- tm_map(corp, removePunctuation)   
corp <- tm_map(corp, removeNumbers)
corp <- tm_map(corp, tolower)
corp <- tm_map(corp, removeWords, stopwords("english"))
corp <- tm_map(corp, stemDocument)
corp <- tm_map(corp, stripWhitespace)  
corp <- tm_map(corp, PlainTextDocument)   

## Create term document matrix
tdm <- TermDocumentMatrix(corp)

## Vector of 1 and 2 indicating recalled and not recalled
trainlabel <- amz_sub$recalled

## Run DNMF
dnmf_output <- DNMF(tdm, trainlabel)

#################
####  Plots  ####
#################

## Plot documents (H matrix) to see if the points fall neatly into two clusters
## (i.e. a recalled and not-recalled cluster) -- per discussion with Valentina
## on 28 July.

## Columns in H matrix are documents; rows are weights of document relative to
## classes (1 = recalled, 2 = not recalled)
colnames(dnmf_output$H) <- trainlabel
rownames(dnmf_output$H) <- c(1, 2)

## Create data frame where each row is a document. orig_class column should have
## the original class (1 or 2 for recalled/not recalled). Columns "val1" and
## "val2" should be the weights of document relative to each class.
h <- data.frame(dnmf_output$H) %>%
  gather(x, value) %>%
  mutate(orig_class = rep(trainlabel, each = 2),
         class = rep(c("val1", "val2"), nrow(.) / 2)) %>%
  spread(class, value)

## Plot val1 and val2 against each other and color points by original document
## class to see if points fall into two groups
ggplot(h, aes(x = val1, y = val2, color = as.factor(orig_class))) +
  geom_point() +
  scale_x_log10() +
  scale_y_log10() +
  theme(legend.position = "bottom") +
  labs(x = "Weight in class 1 (recalled)",
       y = "Weight in class 2 (not recalled)",
       color = "Original document class")

## Somthing is off here...
