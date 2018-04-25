Sentiment Analysis on Claritin Side Effects Twitter Data
----------------

This project utilizes IBM's Db2 Warehouse on Cloud service to analyze a Twitter dataset on user sentiment towards Claritin.  The goal of the analysis is to determine which factors (such as gender or current symptoms) contribute the most towards low user sentiment.  The first component of the analysis involves loading the data into the warehouse and performing exploratory data analysis using SQL.  The rest of the project consists of analyzing the data in R.  Some of the methods include pie charts, bar graphs, logistic regression, word clouds, k-means clustering, and hierarchical clustering.

The full report can be found in the file "Sentiment Analysis.pdf".  It shows a complete breakdown of my analysis, including problem identification, motivation, data loading and verification, data exploration using SQL, data preparation in R, method descriptions, model implementation, visualizations, results analysis, and proposed solutions.  Tables and visualizations are included in three appendices.  Appendix A contains Db2 Warehouse on Cloud images on data load status validation.  Appendix B contains SQL output images.  Appendix C contains R output images.

The dataset "ClaritinSideEffects.csv" was provided by my course instructor at the University of Maryland University College.

The SQL script is included as the file "EDA queries.sql."  Likewise, the R script is included as "sentiment analysis script.R."  The requirements are described in the file "requirements.txt".  Instructions on how to use the program are included as comments in the R file.  After opening the file, please read the instructions carefully before executing the code to ensure that the program functions correctly.