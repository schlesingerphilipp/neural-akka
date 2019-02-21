package main.NeuralNet
import main.data.{Data, FetchActor}
import akka.actor.{Actor, ActorLogging, Props}
import main.NeuralNet.NNActor.{Example, Fit}
import main.data.FetchActor.DispatchFetch
import main.search.GradientSearchExample
object NNActor {
  def props(): Props = Props(new NNActor())
  case class Fit(data: Data)
  case class Example()
}
class NNActor() extends Actor with ActorLogging {
  override def receive: Receive = {
    case Example() => example()
    case Fit(data) => fit(data)
    case _ => () => log.error("Unkwn message")
  }


  def fit(data: Data): Unit = {
    log.info("got example data, fit now")
    GradientSearchExample.run(data)
  }

  def example(): Unit = {
    log.info("received example")
    context.actorOf(FetchActor.props()) ! DispatchFetch()
  }



  override def preStart(): Unit = {
    super.preStart()
    log.debug("started me")
    println("print")
  }
  override def postStop(): Unit = {
    log.debug("stopped me")
    super.postStop()
  }

}
