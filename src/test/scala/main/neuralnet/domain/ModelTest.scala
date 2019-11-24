package main.neuralnet.domain

import main.util.Logger
import org.scalatest._
object ModelTest extends FlatSpec {
}
class ModelTest extends FlatSpec with Matchers {
  implicit val logger: Logger = Logger(Option.empty)
  "DoublePrecision sigma" should "up zero to 0.00001" in {
    val expected = 0 + 1 / Math.pow(10,10)
    DoublePrecision.sigma(0) should equal(expected)
  }
  "DoublePrecision sigma" should "down one to 0.99999" in {
    val expected = 1 - 1 / Math.pow(10,10)
    DoublePrecision.sigma(1) should equal(expected)
  }
  "DoublePrecision sigma" should "leave some valid value as it is" in {
    val expected = 0 + 1 / Math.random()
    DoublePrecision.sigma(expected) should equal(expected)
  }

  "An Summing Function" should " return the sum" in {
    val in = Seq(Value(1.0,1.0), Value(1.0,2.0))
    val expected = 1.0 * 1.0 + 1.0 * 2.0
    val sum = Sum()
    sum.getValue(in) should equal(expected)
  }

  "An Sigma Function" should " return exactly this value" in {
    val in = Seq(Value(1.0,1.0), Value(1.0,2.0))
    val z = 1.0 * 1.0 + 1.0 * 2.0
    val expected = DoublePrecision.sigma(1 / (1 + Math.pow(Math.E, -1 * z)))
    val sigma = Sigma()
    sigma.getValue(in) should equal(expected)
  }
}