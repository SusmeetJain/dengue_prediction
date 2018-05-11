# Deng AI
## predicting the spread of Dengue from climate data

### Objective
Dengue has emerged as one of the major epidemics in current times. There is some evidence available about the correlation of this disease with climate. If we are able to model the number of cases reported historically in a place using the climate data for that place over the same period, It shall increase our accuracy in predicting the spread of this disease. Which in turn shall help not only in being better prepared in our efforts at the individual level but also at the government level for prevention as well as treatment measures.

![alt text](https://community.drivendata.org/uploads/default/original/1X/0f3a28954438c90e1935d61f3f2c23e906feb39a.jpg)

>Each year between 50 and 528 million people are infected and approximately 10,000 to 20,000 die in more than 110 affected countries. Let us explore if it is possible to bring these numbers down with the help of machine learning!


### Data
The data has been obtained from *drivenData.org*, it was released as a part of their competition *DengAI*.
In their own words, driven data hosts "data science competitions to save the world". If you are interested in the social impact data can bring about, you must definitely check out their work.
The basic exploration for this data has been recorded in the ipython notebook **EDA.ipynb**.

### Modelling Approaches

A quick walk-through of different modeling techniques used. Each of them is organized in a different iPython notebook

1. **Simple_Linear_Models.ipynb** :
As a baseline, linear Regression and it's variants Ridge and Lasso are used from their implementations in scikit learn. Ridge and Lasso are linear regression with L1 and L2 regularizations respectively.
2. **XGBoost.ipynb** :
Even the more advanced ensemble models like XGBoost did not significantly improve the performance.
3. **Negative_Binomial_Regression.ipynb** :
It was observed that the data is highly skewed. When count data is negatively skewed, negative binomial regression has been seen to work well and it did outperform all the other models, the *statsmodels* library was used for this implementation.
![alt text](https://github.com/SusmeetJain/dengue_prediction/blob/master/images/skewed_distribution.png)
> Since our labels are so highly skewed, negative binomial regression becomes an ideal choice 
4. **Arima_models.ipynb** :
This is an ongoing work of trying to use Arima modeling for predictions.


### Results from Experiments
This is a brief account of different approaches taken in an attempt to improve the performance on the task and how well or not these worked!

##### Feature Engineering
 - Directly using week number as a feature didn't do good, the month was engineered from the 'week start date' and it did better.
 - Using month without or with one hot encoding is good, but response coding doesn't work.
 - adding the city features as a boolean value has done a good improvement to the model. Although the best results are obtained from using different models for the two cities.
 - Since there is a some gap between the change in climate, proliferation of mosquitoes, there bites and finally the reporting of the disease,  we also tried time shifting our data by 0, 1, 2 and 3 steps, although this looked promising, it actually worsened the performance.

##### Feature Selection
 - Some improvement is also achieved through feature selection using recursive feature elimination
 - Selecting features using select k best (f_regression) isn't good enough), we need to try forward feature selection or PCR

##### Hyperparameter Tuning
 - Both L1 and L2 regularization improve the performance of the linear regression model
 - Tuning the alpha and link parameters in negative binomial regression didn't turn up to be very effective

This is a tiny hack, sometimes the model predicts negative numbers, which is not relevant to this cases so clipping the negative predictions to zeros also slightly helps improve the accuracy!

### Next Steps
 - There is a really good possibility of improving the scores with more data. collecting and aggregating more data for multiple locations and longer durations is a time taking effort, but a strongly suggested route to build on this project further!
 - This is again more relevant in case we have more data, using deep learning, LSTMs to be precise. Recurrent neural networks have outperformed traditional algorithms in most cases where the data is sequential.

### References
 - https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/
 - http://www.mathematica-journal.com/2013/06/negative-binomial-regression/
 - http://dengueforecasting.noaa.gov/
 - https://www.cdc.gov/Dengue/
