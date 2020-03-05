package com.sparkml.UsedCarPricePrediction

import com.sparkml.UsedCarPricePrediction.Driver.spark
import org.apache.spark.sql._
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.tuning._
import org.apache.spark.sql.functions.{ callUDF, lit }
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.functions._
import vegas.sparkExt._
import vegas._

object Regression {

  import spark.implicits._
  var categoricalIndexers: Array[StringIndexer] = _
  var categoricalEncoders: Array[OneHotEncoderEstimator] = _
  var v_assembler: VectorAssembler = _

  def getCleanData(): Unit = {
    var carsDF = spark.read.format("csv")
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("cars.csv")

    for (col <- carsDF.columns) {
      val count = carsDF.filter(carsDF(col).isNull || carsDF(col) === "" || carsDF(col).isNaN).count()
      println(col + " has " + count + " null or empty values")
    }

    Vegas("Shows the relationship between make and price using tick marks.").
      withDataFrame(carsDF).
      encodeX("Make", Nominal).
      encodeY("Price", Quantitative).
      mark(Tick).
      show

    //   Scatter plot for numerical features ["Price", "Mileage", "Year"]
    Vegas().
      withDataFrame(carsDF).
      encodeX("Mileage", Quantitative).
      encodeY("Price", Quantitative).
      mark(Point).
      show

    Vegas().
      withDataFrame(carsDF).
      encodeX("Year", Nominal).
      encodeY("Price", Quantitative).
      mark(Point).
      show

    Vegas().
      withDataFrame(carsDF).
      encodeX("Price", Quantitative).
      encodeY("Mileage", Quantitative).
      mark(Point).
      show

    Vegas().
      withDataFrame(carsDF).
      encodeX("Price", Quantitative).
      encodeY("Year", Ordinal).
      mark(Point).
      show

    Vegas().
      withDataFrame(carsDF).
      encodeX("Year", Ordinal).
      encodeY("Mileage", Quantitative).
      mark(Point).
      show

    Vegas().
      withDataFrame(carsDF).
      encodeX("Mileage", Quantitative).
      encodeY("Year", Ordinal).
      mark(Point).
      show

    println("Checking for plausible values of numerical features") 
    val cols = Array("Price", "Year", "Id", "Mileage")
    val assembler = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("features")

    val df = assembler.transform(carsDF).select("features")
    println(cols.mkString("                    "))
    println(Correlation.corr(df, "features").collect()(0))

    //    Keeping the records for each model between the 20th and 80th percentile of the price.
    var p20 = carsDF.groupBy("Model").agg(callUDF("percentile_approx", $"Price", lit(0.2)).as("p20"))
    var p80 = carsDF.groupBy("Model").agg(callUDF("percentile_approx", $"Price", lit(0.8)).as("p80"))
    var mergeDF = p20.join(p80, "Model")
    carsDF = carsDF.join(mergeDF, "Model")
    carsDF = carsDF.filter(($"Price" > $"p20" && $"Price" <= $"p80"))

    carsDF = carsDF.where("Price>=1500 and Price<=200000")
      .where("year>=2008 and Year<=2017")
      .where("Mileage >=5000 and Mileage<=250000")

    carsDF.coalesce(1).write.mode("overwrite").option("header", "true").csv("cleaned_cars.csv")

  }

  def usedCarPricePrediction(r: String): Unit = {

    var carsDF = spark.read.format("csv")
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("cleaned_cars.csv")

    carsDF = carsDF.withColumnRenamed("Price", "label")

    val categoricalColumns = Array("Make", "Model")
    categoricalIndexers = categoricalColumns
      .map(i => new StringIndexer().setInputCol(i).setOutputCol(i + "Index").setHandleInvalid("skip"))
    categoricalEncoders = categoricalColumns
      .map(e => new OneHotEncoderEstimator().setInputCols(Array(e + "Index")).setOutputCols(Array(e + "Encoded")))

    v_assembler = new VectorAssembler()
      .setInputCols(Array("Year", "Mileage", "MakeEncoded", "ModelEncoded"))
      .setOutputCol("features")

    //    val scaler = new StandardScaler()
    //      .setInputCol("features")
    //      .setOutputCol("scaledFeatures")
    //      .setWithStd(true)
    //      .setWithMean(false)

    //    val scaler = new MinMaxScaler()
    //      .setInputCol("features")
    //      .setOutputCol("scaledFeatures")

    var Array(train, test) = carsDF.randomSplit(Array(.8, .2), 1)

    if(r == "lr"){
      linearRegression(train, test)
    } else if (r == "dt") {
      decisionTreeRegression(train, test)
    } else if (r == "rf"){
      randomForestRegression(train, test)
    } else if (r == "gb"){
      gradientBoostedTreeRegression(train, test)
    } else {
      println(s"""Please select correct regression.""")
    }
  }

  def linearRegression(arr: (Dataset[Row], Dataset[Row])): Unit = {

    val train = arr._1
    val test = arr._2

    val lr = new LinearRegression()
      .setMaxIter(10)

    val pipeline = new Pipeline()
      .setStages(categoricalIndexers ++ categoricalEncoders ++ Array(v_assembler, lr))

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.fitIntercept)
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)
      .setParallelism(2)

    val model = cv.fit(train)
    val predictions = model.transform(test)

    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE):- " + rmse)

    val r2Evaluator = new RegressionEvaluator().setMetricName("r2")

    val r2 = r2Evaluator.evaluate(predictions)
    println("r2Evaluator:- " + r2)

  }

  def decisionTreeRegression(arr: (Dataset[Row], Dataset[Row])): Unit = {

    val train = arr._1
    val test = arr._2

    val dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features")

    val pipeline = new Pipeline()
      .setStages(categoricalIndexers ++ categoricalEncoders ++ Array(v_assembler, dt))

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val paramGrid = new ParamGridBuilder()
      .addGrid(dt.maxDepth, Array(1, 5, 10, 15, 16, 17, 20))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)
      .setParallelism(2)

    val model = cv.fit(train)
    val predictions = model.transform(test)

    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    val r2Evaluator = new RegressionEvaluator().setMetricName("r2")
    println("r2Evaluator:- " + r2Evaluator.evaluate(predictions))

  }

  def randomForestRegression(arr: (Dataset[Row], Dataset[Row])): Unit = {

    val train = arr._1
    val test = arr._2

    val rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features")

    val pipeline = new Pipeline()
      .setStages(categoricalIndexers ++ categoricalEncoders ++ Array(v_assembler, rf))

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxDepth, Array(5, 10, 16, 17, 18))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)
      .setParallelism(2)

    val model = cv.fit(train)
    val predictions = model.transform(test)

    println("Root Mean Squared Error (RMSE) on test data = " + evaluator.evaluate(predictions))

    val r2Evaluator = new RegressionEvaluator().setMetricName("r2")
    println("r2Evaluator:- " + r2Evaluator.evaluate(predictions))

  }

  def gradientBoostedTreeRegression(arr: (Dataset[Row], Dataset[Row])): Unit = {

    val train = arr._1
    val test = arr._2

    val gbtr = new GBTRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setSeed(0)

    val pipeline = new Pipeline()
      .setStages(categoricalIndexers ++ categoricalEncoders ++ Array(v_assembler, gbtr))

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val paramGrid = new ParamGridBuilder()
      .addGrid(gbtr.maxDepth, Array(1, 5, 10, 15, 17))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setParallelism(2)
      .setNumFolds(2)

    val model = cv.fit(train)
    val predictions = model.transform(test)

    println("Root Mean Squared Error (RMSE) on test data = " + evaluator.evaluate(predictions))

    val r2Evaluator = new RegressionEvaluator().setMetricName("r2")
    println("r2Evaluator:- " + r2Evaluator.evaluate(predictions))

  }

}