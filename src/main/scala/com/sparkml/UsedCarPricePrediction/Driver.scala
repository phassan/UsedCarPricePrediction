package com.sparkml.UsedCarPricePrediction

import org.apache.spark.sql._
import org.apache.log4j._

object Driver {

  val spark = SparkSession.builder()
    .appName("SparkML-UCPP")
//    .master("local[*]")
    .getOrCreate()

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    var parm = args(0)

    if (parm == "c") {
      Regression.getCleanData()
    } else if (parm == "lr" || parm == "dt" || parm == "rf" || parm == "gb"){
      Regression.usedCarPricePrediction(parm)
    } else {
      println(s"""Please select correct option.
        c for cleaning data
        lr for Linear Regression
        dt for Decision Tree Regression
        rf for Random Forest Regression
        gb for Gradient Boosted Tree Regression""")
    }
    
    spark.stop()
  }
}