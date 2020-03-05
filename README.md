# Used Car Price Prediction
• This project aims to solve the problem of predicting the price of a used car using Apache Spark's Machine Learning Library (MLlib). 
• Implemented and evaluated various regression techniques, including Linear regression, Decision tree regression, Random forest regression, and Gradient-boosted tree regression on a dataset consisting of approximately 1.2 million records of used car sales in the American market from 1997 to 2018, acquired via scraping on TrueCar.com car sales portal. The dataset(https://www.kaggle.com/jpayne/852k-used-car-listings) is available on Kaggle which has features including Mileage, VIN, Make, Model, Year, State and City. 
• Conducted a comparative performance analysis in terms of RMSE and R-squared score to determine which one of the regression techniques works best with the dataset. 

# Requirements
-	[Apache Hadoop](https://hadoop.apache.org/)
-	[Apache Spark](https://spark.apache.org/) 2.4+

#  How to use it?

   This is a maven-scala project.
	  INSTALLATION:
	  ```
	  cd UsedCarPricePrediction
	  mvn package
	  ```
    
	  EXECUTION:
	  ```
	  	./bin/spark-submit \
 	 	 	--class com.sparkml.UsedCarPricePrediction.Driver \
 	 	    	--master <master-url> \
 	 	    	--deploy-mode <deploy-mode> \
 		    	--conf <key>=<value> \
  		  	prediction-0.1.jar \
          option (c/lr/dt/rf/gb)
          
 Option is one of the following:
        c for cleaning data
        lr for Linear Regression
        dt for Decision Tree Regression
        rf for Random Forest Regression
        gb for Gradient Boosted Tree Regression
