package main
import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import akka.event.LoggingAdapter
import com.typesafe.config.ConfigFactory
import main.NeuralNet.NNActor
import main.NeuralNet.NNActor.Example
import main.data.{ExampleData, FetchActor}

object ExampleApp extends App {
  val config = ConfigFactory.load()
  val system: ActorSystem = ActorSystem(config.getString("akka-system"), config)
  val logger: LoggingAdapter = system.log
  val nn: ActorRef =
    system.actorOf(NNActor.props(), "example-Neuron")
  nn ! Example()
  logger.debug("example on the way")


}
