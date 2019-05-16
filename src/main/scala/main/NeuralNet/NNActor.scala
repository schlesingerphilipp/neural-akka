package main.NeuralNet
import main.data.{Data, FetchActor}
import akka.actor.{Actor, ActorLogging, Props}
import main.NeuralNet.NNActor.{Example, Fit, FitEvo}
import main.data.FetchActor.DispatchFetch
import main.search.{GradientSearchExample, PopulationSearch}
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
    case Example() => example()
    case Fit(data) => fit(data)
    case FitEvo(data) => fitEvolutionary(data)
    case _ => () => error("Unkwn message")
  }


  def fit(data: Data): Unit = {
    info("got example data, fit now")
    GradientSearchExample.run(data)
  }
  def fitEvolutionary(data: Data): Unit = {
    info("got example data, fit now with evolutionary strategy")
    new PopulationSearch(data, 10, 10).fit()
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
