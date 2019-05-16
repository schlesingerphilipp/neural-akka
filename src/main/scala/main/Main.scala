package main
import akka.actor.{ActorRef, ActorSystem}
import com.typesafe.config.ConfigFactory
import main.NeuralNet.NNActor
import main.NeuralNet.NNActor.Example
import main.util.{Logger, Logging}

object ExampleApp extends App with Logging {
  val config = ConfigFactory.load()
  val system: ActorSystem = ActorSystem(config.getString("akka-system"), config)
  val logger = Logger(Option(system.log))
  val nn: ActorRef =
    system.actorOf(NNActor.props(), "example-Neuron")
  nn ! Example()
  debug("example on the way")
}
