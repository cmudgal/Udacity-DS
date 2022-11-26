
Blog: Link to Medium blog https://medium.com/@cmudgal05/tale-of-two-cities-analyze-airbnb-data-for-seattle-and-boston-based-on-crisp-dm-fdd9d0dc8855
Prerequisites:
The following libraries are used for the project-
    -numpy
    -pandas
    -matplotlib
    -seaborn
    -sklearn
    
Purpose:   The purpose of this is to analyse the Seattle/Boston datasets to answer questions that can help in business decision-making. In this CRISP-DM is a structured approach that follows for data analysis. The steps followed are Data cleaning, transforming, and modelling data to find useful insights that are beneficial for business decision-making. 
Business Understanding: Looking at the datasets for Boston and Seattle to answer following questions
1) How does the price vary with time?
2) Which months are busy for booking?
3) How does cancellation policy impact the bookings?

Data Prepration:
1) Filling missing data
2) Removing data
3) Transforming data

Data Modelling:
For predicting the price of listings, prepared data is split into train and test data. The training data is used to fit the linear model. The test data is used to test the linear model.  r-squared score is done to evaluate the model. The price prediction for Seattle has an r-squared score of .52 on test data, which is higher as compared to Boston which has an r-squared score of .19. 

Conclusion: It was found that the busiest time to visit Seattle is in January and for Boston is in September when the bookings were at all time high regardless of the the cancellation policy.
