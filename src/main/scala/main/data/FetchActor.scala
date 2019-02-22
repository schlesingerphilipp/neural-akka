package main.data

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import main.NeuralNet.NNActor.Fit
import main.data.FetchActor.{DispatchFetch, Fetch, FetchExample}

import scala.collection.immutable

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
    ref ! Fit(data)
  }

  def fetch(ref: ActorRef): Unit = log.debug("fetch")

}
object ExampleData {
  def make(seed: Double, factors: Integer, sampleSize: Integer): Seq[DataPoint] = {
    def getXs(): Seq[Double] = {
      for (i <- 0 until factors) yield seed * Math.random()
    }
    def getPoint(ws:Seq[Double]): DataPoint = {
      val xs = getXs()
      val y =  xs.zipAll(ws,0.0,0.0).map((a:(Double,Double)) =>a._1*a._2).foldLeft(0.0)(_ + _)
      DataPoint(y, xs)
    }
    val weights = for (i <- 0 until factors)
      yield Math.random()
    for (j <- 0 until sampleSize)
      yield getPoint(weights)
  }
}
case class ExampleData(seed: Double, factors: Integer, sampleSize: Integer) extends Data {
  val data: Seq[DataPoint] = ExampleData.make(seed, factors, sampleSize)
}
case class DataPoint(target: Double, features: Seq[Double])
sealed trait Data{
  val data: Seq[DataPoint]
  override def toString(): String = {
    val seq: Seq[String] = for {
      i <- data
    } yield i.toString()
    seq.foldLeft("")(_ + " , " +  _)
  }
  def getTrainingTestSplit(trainingPortion: Double): (Seq[DataPoint], Seq[DataPoint]) = {
    data.splitAt(Math.round(data.length * trainingPortion).toInt)
  }
}