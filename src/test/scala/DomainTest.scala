import akka.event.LoggingAdapter
import main.NeuralNet.{Network, _}
import main.data.{Data, DataPoint}
import org.scalatest._
object DomainTest extends FlatSpec {
  implicit val logger: Option[LoggingAdapter] = Option.empty
  def initNet(): Network = {
    val i1 = Node(0, Sum())
    val i2 = Node(1, Sum())
    val h1 = Node(2, Sigma())
    val h2 = Node(3, Sigma())
    val o1 = Node(-1, Sum())
    val w1 = Edge(i1, h1, 0.1)
    val w2 = Edge(i1, h2, 0.2)
    val w3 = Edge(i2, h1, 0.3)
    val w4 = Edge(i2, h2, 0.4)
    val w5 = Edge(h1, o1, 0.5)
    val w6 = Edge(h2, o1, 0.6)
    new Network(Seq(w1,w2,w3,w4,w5,w6), 1)
  }
}
class DomainTest extends FlatSpec with Matchers {
  implicit val logger: Option[LoggingAdapter] = Option.empty
  "DoublePrecision sigma" should "up zero to 0.00001" in {
    val expected = 0 + 1 / Math.pow(10,10)
    DoublePrecision.sigma(0) should equal(expected)
  }
  "DoublePrecision sigma" should "down one to 0.99999" in {
    val expected = 1 - 1 / Math.pow(10,10)
    DoublePrecision.sigma(1) should equal(expected)
  }
  "DoublePrecision sigma" should "leave some valid value alone" in {
    val expected = 0 + 1 / Math.random()
    DoublePrecision.sigma(expected) should equal(expected)
  }

  "An Summing Function" should " return the sum" in {
    val in = Seq((1.0,1.0), (1.0,2.0))
    val expected = 1.0 * 1.0 + 1.0 * 2.0
    val sum = Sum()
    sum.getValue(in) should equal(expected)
  }

  "An Sigma Function" should " return exactly this value" in {
    val in = Seq((1.0,1.0), (1.0,2.0))
    val z = 1.0 * 1.0 + 1.0 * 2.0
    val expected = DoublePrecision.sigma(1 / (1 + Math.pow(Math.E, -1 * z)))
    val sigma = Sigma()
    sigma.getValue(in) should equal(expected)
  }

  "An Input Node with incoming value" should "should return this value"  in {
    val i1 = Node(0,Sum())
    val expected = 123
    i1.getOut(Seq(expected), Seq.empty) should equal(expected)
  }

  "A Sigma Node with incoming values" should "have exactly this out"  in {
    val i1 = Node(0,Sum())
    val testNode: Node = Node(1,Sigma())
    val input = 0.5
    val weightVal = 1
    val weights = Seq[Edge](Edge(i1, testNode, weightVal))
    val z = (input * weightVal)
    val expected = DoublePrecision.sigma(1/ (1 + Math.pow(Math.E, -1 * z)))
    testNode.getOut(Seq(input), weights) should equal(expected)
  }

  "A initial feed-forward" should "have exactly this value" in {
    val i1 = 0.05
    val i2 = 0.1
    val h1 = DoublePrecision.sigma(1/ (1 + Math.pow(Math.E, -1 * (i1 * 0.1 + i2 * 0.3))))
    val h2 = DoublePrecision.sigma(1/ (1 + Math.pow(Math.E, -1 * (i1 * 0.2 + i2 * 0.4))))

    val o1 = h1 * 0.5 + h2 * 0.6
    val net = DomainTest.initNet()
    net.predict(Seq(0.05, 0.1)) should equal(o1)
  }


  "One Learning Step" should "update weights to these values" in {
    val net = DomainTest.initNet()
    val target = 1
    val lRate = 1
    val out = net.predict(Seq(0.05, 0.1))
    val oi1 = net.weights.find(_.from.id.equals(0)).map(_.from).get.out.get
    val oi2 = net.weights.find(_.from.id.equals(1)).map(_.from).get.out.get
    val oh1 = net.weights.find(_.from.id.equals(2)).map(_.from).get.out.get
    val oh2 = net.weights.find(_.from.id.equals(3)).map(_.from).get.out.get
    val dOut = target - out
    val dw6 = dOut * oh2
    val dw5 = dOut * oh1
    val dh1 = oh1*(1- oh1)
    val dh2 = oh2*(1- oh2)
    val dw4 = dw6 * dh2 * oi2
    val dw3 = dw5 * dh1 * oi2
    val dw2 = dw6 * dh2 * oi1
    val dw1 = dw5 * dh1 * oi1
    val w1Expec = 0.1 + lRate * dw1
    val w2Expec = 0.2 + lRate * dw2
    val w3Expec = 0.3 + lRate * dw3
    val w4Expec = 0.4 + lRate * dw4
    val w5Expec = 0.5 + lRate * dw5
    val w6Expec = 0.6 + lRate * dw6
    val newWeights = net.learningStep(1, Seq(0.05, 0.1), net.weights)
    val w1Actual = newWeights.find((x)=>x.from.id.equals(0) && x.to.id.equals(2)).get.weight
    val w2Actual = newWeights.find((x)=>x.from.id.equals(0) && x.to.id.equals(3)).get.weight
    val w3Actual = newWeights.find((x)=>x.from.id.equals(1) && x.to.id.equals(2)).get.weight
    val w4Actual = newWeights.find((x)=>x.from.id.equals(1) && x.to.id.equals(3)).get.weight
    val w5Actual = newWeights.find((x)=>x.from.id.equals(2) && x.to.id.equals(-1)).get.weight
    val w6Actual = newWeights.find((x)=>x.from.id.equals(3) && x.to.id.equals(-1)).get.weight
    assert(w1Actual == w1Expec)
    assert(w2Actual == w2Expec)
    assert(w3Actual == w3Expec)
    assert(w4Actual == w4Expec)
    assert(w5Actual == w5Expec)
    assert(w6Actual == w6Expec)
  }
  "An update step " should "reduce the error" in {
    val net = DomainTest.initNet()
    val before = net.predict(Seq(0.05, 0.1))
    val newWeights = net.learningStep(1, Seq(0.05, 0.1), net.weights)
    val fittedNet = new Network(newWeights, 0.1)
    val after = fittedNet.predict(Seq(0.05, 0.1))
    val didImprove = if (before > 1) after < before else if (before != 1) after > before else true
    assert(didImprove)
  }
}