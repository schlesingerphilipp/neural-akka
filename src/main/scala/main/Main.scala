package main

import java.sql.{Connection, DriverManager}
import java.util.Properties

import main.data.{DataPipeline, SqlFeature, SqlTarget}
import main.neuralnet.domain._
import main.search.StreamingSearch

object Main extends App {
  override def main(args: Array[String]) = {
    nextSearch()
  }

  def nextSearch() = {
    val url = "jdbc:postgresql://localhost/example"
    val props = new Properties()
    props.setProperty("user", "example")
    props.setProperty("password", "example")
    props.setProperty("ssl", "false")
    val conn: Connection = DriverManager.getConnection(url, props)
    val target = SqlTarget("abseth", "abseth_open", "time")
    val features = Seq(SqlFeature("absusd", Seq("absusd_open", "absusd_close"), "time", 0))
    val dp = DataPipeline(target, features, 50)(conn)
    val inputFunctionSelector = (s: SelectorParameters) => Input()
    val hiddenFunctionSelector = (s: SelectorParameters) => Sigma()
    print(dp.numberOfRows)
    val modelDefinition = NextModelParametersDefinition(
      Seq("absusd"), SelectorParameters(),
      inputFunctionSelector,
      SelectorParameters(),
      hiddenFunctionSelector,
      20,
      10,
      20)
    val modelParameters = new NextGenerationService().generateDefinition(modelDefinition)
    val model: NextLayeredNetwork = new NextGenerationService().generateModel(modelParameters)
    val trained = StreamingSearch.trainUntil(model, dp, 0.7, 0)
    val debug = "weed"
  }
}