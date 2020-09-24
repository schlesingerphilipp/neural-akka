package main.neuralnet.domain


trait Function {
  def derivative(out: Double): Double
  def getValue(input: Seq[Value]): Double
}
abstract class Linear extends Function {
  override def derivative(out: Double): Double = 1
}
case class Sum() extends Linear {
  override def getValue(input: Seq[Value]): Double = input.map(i=> i.weight * i.feature).sum
}
case class Input() extends Linear {
  override def getValue(input: Seq[Value]): Double = input.map(_.feature).sum
}

abstract class Logarithmic() extends Function {
  override def derivative(out: Double): Double = out* (1 - out)
}

case class Sigma() extends Logarithmic {
  override def getValue(input: Seq[Value]): Double = {
    val z = input.map(i=> i.weight * i.feature).sum
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
case class Value(weight: Double, feature: Double)
