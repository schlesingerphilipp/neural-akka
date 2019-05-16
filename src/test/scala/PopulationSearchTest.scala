import main.NeuralNet.{Network, _}
import main.data.{ExampleData}
import main.search.{PopulationSearch}
import main.util.Logger
import org.scalatest._

object PopulationSearchTest extends FlatSpec {


}
class PopulationSearchTest extends FlatSpec with Matchers {
  implicit val logger = Logger(Option.empty)

  val inputs = Seq(Node(0, Input()), Node(1, Input()),Node(2, Input()),Node(3, Input()),Node(4, Input()))
  val hidden = Seq(Node(5, Sigma()), Node(6, Sigma()),Node(7, Sigma()),Node(8, Sigma()), Node(9, Sigma()))
  val out = Seq(Node(-1, Sum()))
  val edgesInputHiddenOne: Seq[Edge] = (for (i <- 0 until 5) yield  for (j <- 0 until 5) yield Edge(inputs.apply(i), hidden.apply(j), Math.random())).flatten
  val edgesHiddenOneOut: Seq[Edge] = for (i <- 0 until 5) yield Edge(hidden.apply(i), out.apply(0), Math.random())
  val netTest = new Network(edgesInputHiddenOne ++ edgesHiddenOneOut, 1)


  def getNetwork: Network = {
    PopulationSearch.spawnNet(
      5,1, PopulationSearch.getRandomHiddenLayerVolume(5),
      PopulationSearch.fullWeave, Sigma()).model.asInstanceOf[Network]
  }

  "A Population Search "should " not throw runtime exceptions and should yield a Model" in {
    val data = ExampleData(5,5,5000)
    val search = new PopulationSearch(data, 10, 10)
    val model = search.fit()
    model.getMSE(data) should equal(0) //hahah
  }

  "A Population initialized with size" should "have size" in {
    var expected = 10
    var actual = PopulationSearch.createPopulation(3,10,0).members.size
    actual should equal(expected)
    //Corner case Zero
    expected = 0
    actual = PopulationSearch.createPopulation(3,0,1).members.size
    actual should equal(expected)
    //Corner Case negative size
    expected = 0
    actual = PopulationSearch.createPopulation(3,-10,2).members.size
    actual should equal(expected)
  }
  "A FULL symetric spawn" should "be fully connected between each layer" in {
    val net: Network = getNetwork
    //5 inputs to 5 hidden to 1 out = 5 * 5 + 5 * 1 = 30 edges aka weights
    net.weights.size should equal(30)
    netTest.weights.map(testW =>
      net.weights.exists(e => (e.from.id equals testW.from.id) && e.to.id.equals(testW.to.id)))
      .reduce(_ & _) should equal(true)
  }
  "A full SYMETRIC spawn" should "have an equal amount of nodes in every hidden layer" in {
    val net: Network = getNetwork
    val hiddenOne = net.weights.filter(_.from.layer.get == 1).map(_.from).distinct
    val hiddenTwo = net.weights.filter(_.from.layer.get == 2).map(_.from).distinct
    hiddenOne.length > 0 should equal(true)
    hiddenTwo.length > 0 should equal(true)
    hiddenOne.length should equal(hiddenTwo.length)
  }
  "An interWeave " should "yield Edges" in {
    //TODO

  }
  "A symetricWeave " should "yield a Sequence of node tuples" in {
    //TODO

  }
  "A symetricWeave of 5 nodes from 2 o 4" should " yield 3 Node tuples, where node 2,3,4 are members" in {
    //TODO

  }
  "In a fullSymetricSpawn with 3 hidden layers, 5 inputs and 1 output, there  " should "be 15 * 5 + 1 = 76 Edges " in {
    //TODO

  }
  "A Population initialized with size" should "have size after mixing with other population" in {
    //TODO implement method
  }
  "A PopulationSearch" should "decrease the error" in {
    //TODO implement search
  }
  "A PopulationSearch" should " be able to produce one prediction" in {
    //TODO implement search
  }

}
