-- DATA 650 Assignment 4 
-- Written by Daanish Ahmed
-- Semester Spring 2018
-- Professor Elena Gortcheva
-- April 22, 2018

-- This script is used to perform exploratory data analysis on the Claritin side effects dataset.
-- It involves analyzing the sentiment of patients based on characteristics such as gender and 
-- any associated side effects.




--Displays the total number of rows.
SELECT COUNT(INTERACTION_ID) AS num_rows FROM CLARITIN;




-- Displays the number of rows for each sentiment level (excluding null values).
SELECT sentiment, COUNT(INTERACTION_ID) AS num_rows FROM CLARITIN
WHERE sentiment IS NOT NULL
GROUP BY sentiment
ORDER BY sentiment DESC;




-- Displays the number of male and female patients and the average sentiment for each gender.
SELECT gender, COUNT(INTERACTION_ID) AS num_rows, AVG(sentiment) AS avg_sent FROM CLARITIN
WHERE (gender = 'female' OR gender = 'male') AND sentiment IS NOT NULL
GROUP BY gender
ORDER BY gender;




-- Shows the sentiment count for patients with any of the possible side effects.
SELECT sentiment, COUNT(INTERACTION_ID) AS num_rows FROM CLARITIN
WHERE dizziness = 'yes' OR convulsions = 'yes' OR heart_palp = 'yes' OR short_breath = 'yes' OR headaches = 'yes' 
OR drug_decr = 'yes' OR allergies = 'yes' OR bad_inter = 'yes' OR nausea = 'yes' OR insomnia = 'yes'
GROUP BY sentiment
ORDER BY sentiment DESC;




-- Shows the count and average sentiment for people who are not experiencing any of the possible side effects.
SELECT COUNT(INTERACTION_ID) AS num_rows, AVG(sentiment) AS avg_sent FROM CLARITIN
WHERE dizziness = 'no' AND convulsions = 'no' AND heart_palp = 'no' AND short_breath = 'no' AND headaches = 'no' 
AND drug_decr = 'no' AND allergies = 'no' AND bad_inter = 'no' AND nausea = 'no' AND insomnia = 'no';




-- Shows the count and average sentiment for people who are suffering from at least one potential side effect.
SELECT COUNT(INTERACTION_ID) AS num_rows, AVG(sentiment) AS avg_sent FROM CLARITIN
WHERE dizziness = 'yes' OR convulsions = 'yes' OR heart_palp = 'yes' OR short_breath = 'yes' OR headaches = 'yes' 
OR drug_decr = 'yes' OR allergies = 'yes' OR bad_inter = 'yes' OR nausea = 'yes' OR insomnia = 'yes';




-- Compare average sentiment for people with headaches, shortness of breath, worse allergies, nausea, or insomnia.
SELECT headaches, short_breath, allergies, nausea, insomnia, AVG(sentiment) AS avg_sent FROM CLARITIN
WHERE headaches = 'yes' OR short_breath = 'yes' OR allergies = 'yes' OR nausea = 'yes' OR insomnia = 'yes'
GROUP BY headaches, short_breath, allergies, nausea, insomnia
ORDER BY headaches DESC, short_breath DESC, allergies DESC, nausea DESC, insomnia DESC;




-- Shows the count and percentage of each sentiment for patients with worse allergies after taking Claritin.
SELECT allergies, sentiment, COUNT(INTERACTION_ID) AS num_rows,

	-- Computes percentage for each sentiment level where allergies are worse.
	(COUNT(sentiment) * 100.0 / (SELECT COUNT(allergies) 
	FROM CLARITIN WHERE allergies = 'yes' AND sentiment IS NOT NULL)) AS percent 
	
FROM CLARITIN	
WHERE allergies = 'yes' AND sentiment IS NOT NULL
GROUP BY allergies, sentiment
ORDER BY sentiment DESC;




-- Shows the tweet content, gender, and sentiment level for patients with headaches.
SELECT content, gender, sentiment FROM CLARITIN
WHERE headaches = 'yes' AND content IS NOT NULL
ORDER BY gender, sentiment DESC;
