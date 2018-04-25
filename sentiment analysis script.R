# DATA 650 Assignment 4
# Written by Daanish Ahmed
# Semester Spring 2018
# Professor Elena Gortcheva
# April 22, 2018

# This R script involves performing sentiment analysis on a dataset containing 
# Twitter posts from Claritin users.  The goal is to analyze patient sentiment 
# and study the impact from other factors such as side effects and patient 
# gender.  The script will obtain the dataset from DB2 Warehouse on Cloud 
# database.  The models include a pie chart, bar graphs, logistic regression, 
# word cloud, and both k-means and hierarchical clustering.




# This section involves initializing the packages used in this script.

# Installs the required packages (please only install packages that have not 
# been installed yet, and only install them once).

# Remove the # symbol to install the packages.
# install.packages("ibmdbR")
# install.packages("tm")
# install.packages("SnowballC")
# install.packages("wordcloud")
# install.packages("cluster")

# Loads the packages we need for this assignment.
library(ibmdbR)             # Used to connect to DB2 database
library(tm)                 # Used for text mining
library(SnowballC)          # Used for text mining
library(wordcloud)          # Used for word clouds
library(ggplot2)            # Used for bar plots
library(cluster)            # Used for k-means clustering

# End of initializing packages.




# This section involves accessing the DB2 Warehouse on Cloud database server.

# The following lines are used to access the server and include credentials such as 
# the host name, user ID and password.
dsn_driver <- c("BLUDB")
dsn_database <- c("BLUDB")
dsn_hostname <- c("dashdb-entry-yp-dal09-08.services.dal.bluemix.net")
dsn_port <- "50000"
dsn_protocol <- "TCPIP"
dsn_uid <- c("dash15475")
dsn_pwd <- c("503043b9b405")
conn_path <- paste(dsn_driver,  
                   ";DATABASE=",dsn_database,
                   ";HOSTNAME=",dsn_hostname,
                   ";PORT=",dsn_port,
                   ";PROTOCOL=",dsn_protocol,
                   ";UID=",dsn_uid,
                   ";PWD=",dsn_pwd,sep="")
mycon <- idaConnect(conn_path) 
idaInit(mycon)

# End of loading the database server.




# This section involves loading the data into a data frame and verifying that 
# it was loaded correctly.

# Creates a new data frame to store the Claritin data from the database.
CLAR_SE <- idaQuery("SELECT * from CLARITIN")

# Checks the row counts in the new data frame and the database.
nrow(CLAR_SE)
idadf(mycon, "SELECT COUNT(*) FROM CLARITIN")

# Lists all tables in the database.
idaShowTables()

# Verifies that the Claritin table exists.
idaExistTable('CLARITIN')

# Shows the number of tweets per sentiment.
table(CLAR_SE$SENTIMENT)
idadf(mycon, "SELECT sentiment, 
      COUNT(1) AS count 
      FROM CLARITIN 
      WHERE sentiment IS NOT NULL
      GROUP BY sentiment
      ORDER BY sentiment DESC")

# End of loading and validating the data.




# This section of code covers data preprocessing.

# Previews the dataset.
View(CLAR_SE)

# The first two variables are unique values and are not useful for the analysis, 
# so we remove them.
CLAR_SE$INTERACTION_ID <- NULL
CLAR_SE$ARTICLE_URL <- NULL

# This function checks to see how many missing values are in each variable.
apply(CLAR_SE, 2, function(CLAR_SE) sum(is.na(CLAR_SE)))

# Displays the rows with missing values.
CLAR_SE[!complete.cases(CLAR_SE),]

# Removes rows with missing values.
CLAR_SE <- na.omit(CLAR_SE)

# Verifies that the missing values have been removed.
apply(CLAR_SE, 2, function(CLAR_SE) sum(is.na(CLAR_SE)))

# Convert the following character variables into factors.  This step is useful 
# for certain methods such as the pie charts and bar graphs.
CLAR_SE$RELEVANT <- as.factor(CLAR_SE$RELEVANT)
CLAR_SE$SENTIMENT <- as.factor(CLAR_SE$SENTIMENT)
CLAR_SE$GENDER <- as.factor(CLAR_SE$GENDER)
CLAR_SE$DIZZINESS <- as.factor(CLAR_SE$DIZZINESS)
CLAR_SE$CONVULSIONS <- as.factor(CLAR_SE$CONVULSIONS)
CLAR_SE$HEART_PALP <- as.factor(CLAR_SE$HEART_PALP)
CLAR_SE$SHORT_BREATH <- as.factor(CLAR_SE$SHORT_BREATH)
CLAR_SE$HEADACHES <- as.factor(CLAR_SE$HEADACHES)
CLAR_SE$DRUG_DECR <- as.factor(CLAR_SE$DRUG_DECR)
CLAR_SE$ALLERGIES <- as.factor(CLAR_SE$ALLERGIES)
CLAR_SE$BAD_INTER <- as.factor(CLAR_SE$BAD_INTER)
CLAR_SE$NAUSEA <- as.factor(CLAR_SE$NAUSEA)
CLAR_SE$INSOMNIA <- as.factor(CLAR_SE$INSOMNIA)

# Verifies that the variables have been converted.
summary(CLAR_SE)

# Removes non-English tweets and bad gender information.
CLAR_SE <- CLAR_SE[CLAR_SE$RELEVANT != "non_english", ]
CLAR_SE <- CLAR_SE[CLAR_SE$GENDER != "bad_link_or_company", ]

# Remove "relevant" and "convulsions," since they only have 1 level.
CLAR_SE$RELEVANT <- NULL
CLAR_SE$CONVULSIONS <- NULL

# Verifies that the variables have been removed.
summary(CLAR_SE)

# Creates a new variable "symptoms" that lists the user side effects.
CLAR_SE$SYMPTOM <- ifelse(CLAR_SE$DIZZINESS == "yes", "dizziness", "none")
CLAR_SE$SYMPTOM <- ifelse(CLAR_SE$HEART_PALP == "yes", "heart palpitations", CLAR_SE$SYMPTOM)
CLAR_SE$SYMPTOM <- ifelse(CLAR_SE$SHORT_BREATH == "yes", "shortness of breath", CLAR_SE$SYMPTOM)
CLAR_SE$SYMPTOM <- ifelse(CLAR_SE$HEADACHES == "yes", "headaches", CLAR_SE$SYMPTOM)
CLAR_SE$SYMPTOM <- ifelse(CLAR_SE$DRUG_DECR == "yes", "drug effect decreased", CLAR_SE$SYMPTOM)
CLAR_SE$SYMPTOM <- ifelse(CLAR_SE$ALLERGIES == "yes", "allergies", CLAR_SE$SYMPTOM)
CLAR_SE$SYMPTOM <- ifelse(CLAR_SE$BAD_INTER == "yes", "bad interaction", CLAR_SE$SYMPTOM)
CLAR_SE$SYMPTOM <- ifelse(CLAR_SE$NAUSEA == "yes", "nausea", CLAR_SE$SYMPTOM)
CLAR_SE$SYMPTOM <- ifelse(CLAR_SE$INSOMNIA == "yes", "insomnia", CLAR_SE$SYMPTOM)

# Converts the "symptoms" variable into a factor.
CLAR_SE$SYMPTOM <- as.factor(CLAR_SE$SYMPTOM)

# Verifies the levels of the "symptoms" variable.
summary(CLAR_SE$SYMPTOM)

# End of data preprocessing.




# This section involves building a pie chart showing the frequency of each side effect.

# Creates a copy of the Claritin data frame excluding patients with no symptoms.
CLAR_SE_2 <- CLAR_SE[CLAR_SE$SYMPTOM != "none", ]

# Resets the margin of the graph.
par(mar=c(1,1,1,1))

# Builds the pie chart and sets the colors of each label.
pie(table(CLAR_SE_2$SYMPTOM), col=c("yellow", "blue", "green", "red", "purple", "pink", "orange", 
                                    "cyan", "black", "magenta"))

# End of creating pie chart.




# This section involves creating bar graphs to compare the patient sentiment, gender, 
# and observed side effects.

# Builds a bar chart showing the distribution of side effects per sentiment level.
ggplot(CLAR_SE_2, aes(CLAR_SE_2$SENTIMENT, fill=SYMPTOM)) + geom_bar()

# Builds a bar chart showing the patients' genders for each side effect.
ggplot(CLAR_SE_2, aes(CLAR_SE_2$SYMPTOM, fill=GENDER)) + geom_bar()

# End of creating bar charts.




# This section involves building a logistic regression model to predict the patient 
# sentiment.  It will predict whether the sentiment will equal "1" (negative) or "0" 
# (positive or neutral).

# Creates a copy of the dataset that omits variables not needed for this model.
CLAR_SE_LR <- CLAR_SE[, -c(1, 2)]
CLAR_SE_LR$SYMPTOM <- NULL

# Restructures the target variable "sentiment" as a binary factor variable with 
# two values: "1" means a negative sentiment of 1 or 2, and "0" means a positive 
# or neutral sentiment of 3, 4, or 5.
CLAR_SE_LR$SENTIMENT <- ifelse(CLAR_SE$SENTIMENT %in% c("1", "2"), "1", "0")
CLAR_SE_LR$SENTIMENT <- as.factor(CLAR_SE_LR$SENTIMENT)

# Generates a random seed to reproduce the results.
set.seed(12345)

# Partitions the Claritin dataset into a training set with 70% of the data and a 
# test set with 30% of the data.
ind <- sample(2, nrow(CLAR_SE_LR), replace = TRUE, prob = c(0.7, 0.3))
train.data <- CLAR_SE_LR [ind == 1, ]
test.data <- CLAR_SE_LR [ind == 2, ]

# Builds the logistic regression model on the training set.
model <- glm(SENTIMENT ~ ., family = binomial, data = train.data)

# Creates a confusion matrix for the training data.
train_pred <- round(predict(model, train.data, type="response"))
table(train_pred, train.data$SENTIMENT)

# Creates a confusion matrix for the test data.
test_pred <- round(predict(model, test.data, type="response"))
table(test_pred, test.data$SENTIMENT)

# End of building the logistic regression model.




# This section involves text mining preprocessing to analyze the content of the 
# tweets.  It will prepare the data for word cloud analysis.

# Creates a data frame that only contains tweets with negative sentiment (1 or 2).
negative <- idadf(mycon, "SELECT CONTENT FROM CLARITIN 
                  WHERE SENTIMENT = '1' OR SENTIMENT = '2'")

# Builds the tweets corpus.
tweets <- VectorSource(negative$CONTENT)
tweets <- Corpus(tweets)

# Examines one of the tweets' content.
inspect(tweets[[10]])

# These commands remove any URLs that may exist in these documents.
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
tweets <- tm_map(tweets, content_transformer(removeURL))

# Removes all numbers and punctuation.
tweets = tm_map(tweets, removeNumbers)
tweets = tm_map(tweets, removePunctuation)

# This is a list of additional stop words and unnecessary words that were not included 
# in the default stopwords lists.
stop = c("just", "good", "watch", "time", "join", "get", "big", "going", "much", "said", 
         "like", "will", "now", "new", "can", "ass", "doesnt", "gave", "means", "one", 
         "mr", "less", "from", "looking", "ago", "come", "sat", "cut", "must", "full", 
         "im", "make", "fuck", "next", "give", "let", "shit", "thing", "weve", "back", "dont", 
         "let", "meet", "begin", "bring", "make", "set", "stay", "send", "step", "stop", 
         "open", "ask", "hold", "come", "wont", "run", "seek", "hear", "lot", "theyre", 
         "their", "i", "ive", "put", "didnt", "try", "tri", "claritin", "amp", "becaus", 
         "lol", "gonna", "wanna", "alway", "isnt", "suck", "damn", "smh", "happen", "made", 
         "ani", "probabl", "aint", "suppos", "realli", "noth", "befor", "jonathanrknight", 
         "whi", "alreadi", "onli", "anymor", "sinc")

# Removes stop words and unimportant words.  It identifies stop words from the English 
# and Smart stopwords lists, as well as words included in the 'stop' variable.
tweets = tm_map(tweets, removeWords, c("the", "and", stop, stopwords("english"), 
                                       stopwords("SMART")))

#Removes special characters such as @, â, and the Euro symbol.
toSpace <- content_transformer(function (x, pattern) gsub(pattern, " ", x))
tweets <- tm_map(tweets, toSpace, "@")
tweets <- tm_map(tweets, toSpace, "â")
tweets <- tm_map(tweets, toSpace, "/")
tweets <- tm_map(tweets, toSpace, "\\|")            # Removes the "|" character
tweets <- tm_map(tweets, toSpace, "\n")             # Removes new line character
tweets <- tm_map(tweets, toSpace, "\u20ac")         # Removes the Euro symbol
tweets <- tm_map(tweets, toSpace, "\u201d")         # Removes the " symbol

# Removes non-ASCII characters.
removeInvalid <- function(x) gsub("[^\x01-\x7F]", "", x)
tweets <- tm_map(tweets, content_transformer(removeInvalid))

# Performs stemming on each word, reducing it to its root word.
tweets <- tm_map(tweets, stemDocument)

# Changes all letters to lowercase.
tweets = tm_map(tweets, content_transformer(tolower))

# Removes stop words again, since some stop words may appear after stemming.
tweets = tm_map(tweets, removeWords, c("the", "and", stop, stopwords("english"), 
                                       stopwords("SMART")))

# Removes all extra whitespace between words.
tweets =  tm_map(tweets, stripWhitespace)

# Verifies that the tweet content has been preprocessed.
inspect(tweets[10])

# Builds the Document Term Matrix (DTM) using the tweet content.
tweet_dtm <- DocumentTermMatrix(tweets)

# End of text preprocessing.




# This section covers the creation of word clouds to visualize the most frequent 
# terms appearing in the Claritin tweets.

# Creates a list of all unique words and their frequency counts.
freq <- colSums(as.matrix(tweet_dtm))

# Color scheme using up to 6 different colors for words depending on their frequency.
dark2 <- brewer.pal(6, "Dark2")

# Builds a word cloud that colors terms according to their frequency.  It has a 
# maximum of 60 words.
set.seed(12345)           # Random seed to reproduce the results.
wordcloud(names(freq), freq, max.words=60, rot.per=0.2, colors=dark2)

# End of generating word clouds.




# This section involves using k-means clustering and hierarchical clustering to 
# categorize word frequencies.

# Removes sparse terms from the DTM.
tweet_dtm_2 <- removeSparseTerms(tweet_dtm, 0.985)

# Shows the properties of the DTM after removing sparse terms.
tweet_dtm_2

# Builds the dissimilarity matrix using the DTM.
dsm <- dist(t(tweet_dtm_2), method="euclidian")

# The following code uses the elbow method to determine the ideal k-value.

# Sets the margins for the elbow method plot.
par(mar=c(4, 4, 4, 4))

# Plots the between clusters sum-of-squares by k value.
bss <- integer(length(2:15))
for (i in 2:15) bss[i] <- kmeans(dsm, centers=i)$betweenss
plot(1:15, bss, type="b", xlab="Number of Clusters",
     ylab="Sum of squares", col="blue")

# Plots the within clusters sum-of-squares by k value.
wss <- integer(length(2:15))
for (i in 2:15) wss[i] <- kmeans(dsm, centers=i)$tot.withinss
lines(1:15, wss, type="b")

# The plot suggests that there are more than one "elbows," meaning that the elbow 
# method is not an ideal method to use.  The alternate approach is to use the 
# formula sqrt(n/2), where n is the number of documents (39).  This results in 
# a k-value of 4.

# Builds the k-means clustering model using k=4 clusters.
kfit <- kmeans(dsm, 4)

# Shows the properties of the clustering model.
kfit

# Displays the new cluster plot.
clusplot(as.matrix(dsm), kfit$cluster, color=T, shade=T, labels=2, lines=0)

# Creates and plots the hierarchical cluster dendrogram.
fit <- hclust(d=dsm, method="ward.D2")
plot(fit, hang=-1)

# The following commands add red boxes to outline the clusters using 
# k=4 clusters.
plot.new()
plot(fit, hang=-1)
groups <- cutree(fit, k=4)
rect.hclust(fit, k=4, border="red")

# End of building the dendrogram.




# End of script.
