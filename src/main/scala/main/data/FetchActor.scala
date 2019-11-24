package main.data

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import main.data.FetchActor.{DispatchFetch, Fetch, FetchExample}

object FetchActor {
  def props(): Props = Props(new FetchActor())
  final case class DispatchFetch()
  final case class FetchExample(target: ActorRef)
  final case class Fetch(target: ActorRef)

}
class FetchActor  extends Actor with ActorLogging{


  override def receive: Receive = {
    case DispatchFetch() => dispatch()
    case FetchExample(ref) => fetchExample(ref)
    case Fetch(ref) => fetch(ref)
  }

  def dispatch(): Unit = {
    log.debug("dispatch")
    context.actorOf(FetchActor.props()) ! FetchExample(sender())
  }

  def fetchExample(ref: ActorRef): Unit = {
    log.debug("fetch Example")
    val data = ExampleData(Math.random() * 10, 3, 100)
  }

  def fetch(ref: ActorRef): Unit = log.debug("fetch")

}

