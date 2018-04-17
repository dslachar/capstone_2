#### U.S. State Department Travel Advisory Classification Prediction Model

* Use published U.S. State Department Travel Advisories to train a NLP model to predict classification. USDS overhauled Travel Advisory system in January 2018.

## Pros

* Publicly available data
* consistent format/plain language
* simple classification targets, Level 2, 3, 4

## Cons
* Limited dataset because new system initiated in January 2018, 
* Formulatic writing in corpus may present challenges:
	* Possibly easy to classify due to certain key words used in each class.
	* Possibly hard to classify due to overlaps in formulatic writing style.

## Challenges

* Tons of processing
	* There will need to be a lot of stop text.
	* Will need to experiment with stop text.
	* Certain formulaic portions of each text will need to either be removed 		from text or added to stop text. 

Is there enough data? 
	~ 115 class 2 class - 4 countries
	~ 200-300 words per advisory
						  
##Current Classifications

Level 1 - 126 - standard format, effect on model

Level 2 - 68

Level 3 - 31

Level 4 - 16

Level 2-4 - 115

##Motivation

Examine geopolitical biases in classification sytem
Create a shell given a set of features
Create a bank of words to use for severity level and type of risk
evaluate consistency

Examine effect of travel advisories on travel

						  
 
	
	 


