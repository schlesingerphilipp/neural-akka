package main.search

import akka.actor.ActorSystem
import akka.event.LoggingAdapter
import com.typesafe.config.ConfigFactory
import main.NeuralNet._
import main.data.Data

object GradientSearchExample {

  val config = ConfigFactory.load()
  val system: ActorSystem = ActorSystem(config.getString("akka-system"), config)
  implicit val logger: Option[LoggingAdapter] = Option(system.log)
  def run(data: Data): Unit = {
    /*

                 node(-1)
               /    |    \
        node(3)  node(4)  node(5)
           |  // \  |   /  \\ |
        node(0)  node(1)  node(2)
     */
    val nodes = Seq(
      Node(0, Sum()),
      Node(1, Sum()),
      Node(2, Sum()),
      Node(3, Sigma()),
      Node(4, Sigma()),
      Node(5, Sigma()),
      Node(-1, Sum()),
    )
    val weights = Seq(
      Edge(nodes.apply(0), nodes.apply(3), Math.random()),
      Edge(nodes.apply(0), nodes.apply(4), Math.random()),
      Edge(nodes.apply(0), nodes.apply(5), Math.random()),
      Edge(nodes.apply(1), nodes.apply(3), Math.random()),
      Edge(nodes.apply(1), nodes.apply(4), Math.random()),
      Edge(nodes.apply(1), nodes.apply(5), Math.random()),
      Edge(nodes.apply(2), nodes.apply(3), Math.random()),
      Edge(nodes.apply(2), nodes.apply(4), Math.random()),
      Edge(nodes.apply(2), nodes.apply(5), Math.random()),
      Edge(nodes.apply(3), nodes.apply(6), Math.random()),
      Edge(nodes.apply(4), nodes.apply(6), Math.random()),
      Edge(nodes.apply(5), nodes.apply(6), Math.random())
    )
    val net: Network = new Network(weights, 1)
    val randomMeanSquareError = net.getMSE(data)
    net.train(data)
    val meanSquareError = net.getMSE(data)
    println("meanSquareError: ", meanSquareError)
    println("improvement: ", (randomMeanSquareError - meanSquareError) / randomMeanSquareError, " parts better than random")
  }

}