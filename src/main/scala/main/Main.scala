package main
import akka.actor.{ActorRef, ActorSystem}
import com.typesafe.config.ConfigFactory
import main.neuralnet.akka.NNActor
import main.util.{Logger, Logging}

object ExampleApp extends App with Logging {
  val config = ConfigFactory.load()
  val system: ActorSystem = ActorSystem(config.getString("akka-system"), config)
  val logger = Logger(Option(system.log))
  val nn: ActorRef =
    system.actorOf(NNActor.props(), "example-Neuron")
  debug("example on the way")
}
