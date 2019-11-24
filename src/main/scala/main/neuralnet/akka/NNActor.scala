package main.neuralnet.akka

import akka.actor.{Actor, ActorLogging, Props}
import main.data.FetchActor.DispatchFetch
import main.data.{Data, FetchActor}
import main.search.{GradientSearchExample}
import main.util.{Logger, Logging}
object NNActor {
  def props(): Props = Props(new NNActor())
  case class Fit(data: Data)
  case class FitEvo(data: Data)
  case class Example()
}
class NNActor() extends Actor with Logging with ActorLogging {
  implicit val l = Logger(Option(log))
  override val logger = l //hmmm ...
  override def receive: Receive = {
    case _ => () => error("Unkwn message")
  }


  def fit(data: Data): Unit = {
    info("got example data, fit now")
    GradientSearchExample.run(data)
  }
  def fitEvolutionary(data: Data): Unit = {
    info("got example data, fit now with evolutionary strategy")
  }

  def example(): Unit = {
    info("received example")
    context.actorOf(FetchActor.props()) ! DispatchFetch()
  }



  override def preStart(): Unit = {
    super.preStart()
    debug("started me")
    println("print")
  }
  override def postStop(): Unit = {
    debug("stopped me")
    super.postStop()
  }

}
