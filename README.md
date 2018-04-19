# Natural Language Processing of US Foreign Aid Descriptions

The United States Agency for International Development (USAID) reports data on US foreign aid projects in accordance with standards established by the Organization for Economic Co-operation and Development (OECD). Member countries self-report project data to the OECD.  In 2017 alone, USAID reported more than 20,000 foreign aid projects. Can Natural Language Processing be used to gain insight on large datasets of US foreign aid? 

<img alt="OECD logo" src="images/oecd_seal.png" width='300'>  <img alt="USAID logo" src="images/usaid_seal.png" width='300'>  


## Table of Contents
1. [Dataset](#dataset)
2. [Setup](setup)
    * [Categories](#categories)
    * [Data Processing](#data-processing)
    * [Segmentation](#segmentation)
3. [Modeling](#modeling)
4. [Modeling](#modeling)
5. [Insights](#insights)    
6. [Future Directions](#future-directions)

## Dataset

USAID provides foreign aid project data at [USAID Data Querry](https://explorer.usaid.gov/query). There are over 467,000 records available dating to 1946.  

<img alt="OECD logo" src="images/data.png" width='700'> 


## Setup

### corpus - 319,974 foreign aid project descriptions (projects between 1999 and 2018)
### targets - category classification of foreign aid projects in corpus

## Categories

<img alt="Category Counts Plot" src="images/category_counts_plot.png" width='600'>

### Data Processing ([code](https://github.com/dslachar/capstone_2))

* Created main data frame with all records from 1999 - 2018
* Retained the following fields:
	* Fiscal Year
	* Activity Description
	* Assistance Category 	
* Removed records with the following Assistance Categories: 
	* Administrative Costs 	
	* Other 
* Removed records with null values (4 null values)
*  Test/Train Split

#### Text Transform

* Pipeline to transform text
	* CountVectoizer
	* Tfidf Transform 	 	 




## Modeling([code](https://github.com/dslachar/capstone_2))

#### Fit the test data to two different models:
	
	* SKLearn Multinomial Naive Bayes
	* SKLearn Linear Support Vector Machine

#### Multinomial Naive Bayes achieves a slightly better accuracy score.

	* Naive Bayes Accuracy Score: 0.852 (95% CI 0.849 - 0.854)
	* Linear SVM Accuracy Score: 0.839

#### Insights

| Category | TPR |
| ------- | -----|
|Agriculture| 0.760 |
|Commodity Assistance| 0.847|
|**Economic Growth**| **0.702** |
|**Education**| **0.667** |
|Governance| 0.924 |
|Health and Population| 0.888 |
|Humanitarian| 0.849 |
|Infarstructure| 0.750 |


## To MVP

* Need to look for patterns in false positives in Education and Economic Growth
* print common words and phrases from each category


## References

   