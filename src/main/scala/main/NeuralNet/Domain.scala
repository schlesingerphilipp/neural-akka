package main.NeuralNet

import java.lang.StackWalker

import akka.event.LoggingAdapter
import main.data.{Data, DataPoint}

import scala.annotation.tailrec


trait Logging {
  val logger: Option[LoggingAdapter]
  def info(m: String): Unit = {
    if (logger.isDefined) {
      logger.get.info(m)
    }
  }
  def debug(m: String): Unit = {
    if (logger.isDefined) {
      logger.get.info(m)
    }
  }
  def warn(m: String): Unit = {
    if (logger.isDefined) {
      logger.get.info(m)
    }
  }
  def error(m: String): Unit = {
    if (logger.isDefined) {
      logger.get.info(m)
    }
  }
}
trait Model {
  def predict(input: Seq[Double]): Double
  def train(data: Data, minimumImprovementThreshold: Double): Model
  def getMSE(data: Data): Double
}


class Network(val weights: Seq[Edge], var lRate: Double)(implicit loggerr: Option[LoggingAdapter]) extends Model with Logging {
  override val logger: Option[LoggingAdapter] = loggerr
  override def predict(input: Seq[Double]): Double = {
    weights.foreach(_.prepare())
    val out: Option[Node] = weights.find(_.to.id == -1).map(_.to)//'out' has id -1
    if (out.isDefined) {
      out.get.getOut(input, weights)
    } else {networkError()}
  }
  def networkError(): Double = {
    error("Network error. Can not find 'out' node under id [-1]")
    0
  }

  def learningStep(target: Double, input: Seq[Double], weights: Seq[Edge]): Seq[Edge] = {
    weights.foreach(_.prepare())//weights are stateful, else you do the same calculation the power of weights times
    updatedWeights(input, target, weights)
  }

  @tailrec
  private def learnFromData(data: Seq[DataPoint], weights: Seq[Edge]): Seq[Edge] = {
    if (data.isEmpty) {
      return weights
    }
    learnFromData(data.tail, learningStep(data.head.target, data.head.features, weights))
  }

  @tailrec
  private def trainRec(data: Data, minimumImprovementThreshold: Double, lastImpro: Double, lastMse: Double,
                       counter: Int, weights: Seq[Edge]): Seq[Edge]= {
    if (counter > 1000 || lastImpro < minimumImprovementThreshold) {
      return weights
    }
    val nextMse = getMSE(data)
    val nextImpro = (lastMse - nextMse) / lastMse
    debug(s"Improvement: ${nextImpro}")
    trainRec(data, minimumImprovementThreshold, nextImpro, nextMse, counter + 1, learnFromData(data.data, weights))
  }

  override def train(data: Data, minimumImprovementThreshold: Double): Model= {
    new Network(trainRec(data, minimumImprovementThreshold, 0, Double.MaxValue, 0, this.weights), lRate)
  }

  private def updatedWeights(input:Seq[Double], target: Double, weights: Seq[Edge]): Seq[Edge] = {
    weights.map(e => {
      val delta = e.from.getOut(input, weights) * e.to.deltaE(weights, input, target)
      val newWeight = e.weight + lRate * delta
      Edge(e.from, e.to, newWeight)
    })
  }

  override def getMSE(data: Data): Double = {
    val mse = data.data.map(e => {
      val prediction = predict(e.features)
      (e.target - prediction) * (e.target - prediction)
    }).sum / data.data.length
    mse
  }
}

case class Edge(from: Node, to: Node, weight: Double) {
  def prepare(): Unit = {
    from.prepare()
    to.prepare()
  }
}




case class Node(id: Int, function: Function, layer: Option[Int] = Option.empty)(implicit loggerr: Option[LoggingAdapter]) extends Logging{
  override val logger: Option[LoggingAdapter] = loggerr
  var in:  Option[Seq[(Double, Double)]] = Option.empty
  var out: Option[Double] =  Option.empty
  var delta: Option[Double] = Option.empty

  def getOut(input: Seq[Double], weights: Seq[Edge]): Double = {
    if (this.out.isDefined) return this.out.get
    val weightsTo = weights.filter(_.to.eq(this))
    if (weightsTo.isEmpty) {
      //Nodes without an incoming edge are input nodes, which take the input value as is
      this.out = Option(input.apply(id))
      return out.get
    }
    this.out = Option(function.getValue(getIn(weightsTo, input, weights)))
    out.get
  }

  def getIn(weightsTo: Seq[Edge], input: Seq[Double], weights: Seq[Edge]): Seq[(Double, Double)] = {
    if (this.in.isDefined) return in.get
    this.in = Option(weightsTo.sortWith(_.from.id < _.from.id)
      .map(e=>(e.from.getOut(input, weights), e.weight)))
    this.in.get
  }


  def deltaE(weights: Seq[Edge] , input: Seq[Double], target: Double): Double = {
    if (delta.isDefined) return delta.get
    val out = getOut(input, weights)
    if (this.id == -1) {
      return target - out
    }
    //delta of a node are all derivatives wrt to nodes value. Deduction of the weight is always 1
    val inEdgeDelta = weights.filter(_.from.equals(this)).map(_.to.deltaE(weights, input, target)).reduce(_ * _)
    this.delta = Option(function.derivative(out) * out *  inEdgeDelta)
    this.delta.get
  }
  def prepare() :Unit = {
    in = Option.empty
    out = Option.empty
    delta = Option.empty
  }


}

trait Function {
  def derivative(out: Double): Double
  def getValue(input: Seq[(Double, Double)]): Double
}
case class Sum() extends Function {
  override def derivative(out: Double): Double = 1
  override def getValue(input: Seq[(Double, Double)]): Double = input.map(i=> i._1 * i._2).sum
}
case class Input() extends Function {
  override def derivative(out: Double): Double = 1
  override def getValue(input: Seq[(Double, Double)]): Double = ??? //Input value is taken from inputs. Do not implement this
}

case class Sigma() extends Function {
  override def derivative(out: Double): Double = out* (1 - out)
  override def getValue(input: Seq[(Double, Double)]): Double = {
    val z = input.map(i=> i._1 * i._2).sum
    DoublePrecision.sigma(1 / (1+ Math.pow(Math.E, -1 * z)))
  }
}
object DoublePrecision {
  //Sigma is value between 1 and 0, but never 1 or 0
  def sigma(o: Double): Double = {
    o match {
      case 1.0 => 1 - 1 / Math.pow(10, 10)
      case 0.0 => 0 + 1 / Math.pow(10, 10)
      case _ => o
    }
  }
}