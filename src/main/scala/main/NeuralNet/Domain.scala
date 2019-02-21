package main.NeuralNet

import java.lang.StackWalker

import akka.event.LoggingAdapter
import main.data.Data


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

class Network(var weights: Seq[Edge], var lRate: Double)(implicit loggerr: Option[LoggingAdapter]) extends Logging {
  override val logger: Option[LoggingAdapter] = loggerr
  def predict(input: Seq[Double]): Double = {
    val out: Option[Node] = weights.find(_.to.id == -1).map(_.to)//'out' has id -1
    if (out.isDefined) {
      out.get.getOut(input, weights)
    } else {networkError()}
  }
  def networkError(): Double = {
    error("Network error. Can not find 'out' node under id [-1]")
    0
  }

  def learningStep(target: Double, input: Seq[Double]): Unit = {
    this.weights.foreach(_.prepare())
    this.weights = updatedWeights(input, target)
  }

  def train(data: Data): Unit = {
    var mse = getMSE(data)
    var previous = -1.0
    var counter = 0
    while (previous == -1.0 || previous < mse || ((previous - mse) / previous) > 0.000001) {
      counter += 1
      data.data.foreach(d => learningStep(d.target, d.features))
      previous = mse
      mse = getMSE(data)
      debug(s"Iteration ${counter}")
      if (previous < mse) {
        info("It got worse")
      } else {
        info(s"Improvement: ${((previous - mse) / previous)}")
      }
      if (counter > 1000) {
        throw new RuntimeException("Exceeded number of iterations")
      }
    }
  }

  def updatedWeights(input:Seq[Double], target: Double): Seq[Edge] = {
    weights.map(e => {
      val delta = e.from.getOut(input, weights) * e.to.deltaE(weights, input, target)
      val newWeight = e.weight - lRate * delta
      Edge(e.from, e.to, newWeight)
    })
  }

  def getMSE(data: Data): Double = {
    val mse = data.data.map(e => {
      val prediction = predict(e.features)
      (e.target - prediction) * (e.target - prediction)
    }).sum / data.data.length
    debug(s"MSE ${mse}")
    mse
  }
}

case class Edge(from: Node, to: Node, weight: Double) {
  def prepare(): Unit = {
    from.prepare()
    to.prepare()
  }
}




case class Node(id: Int, function: Function)(implicit loggerr: Option[LoggingAdapter]) extends Logging{
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