import main.NeuralNet.{Network, _}
import main.data.ExampleData
import main.search._
import main.util.{Logger, TestAdapter}
import org.scalatest._

object PopulationSearchTest extends FlatSpec {


}
class PopulationSearchTest extends FlatSpec with Matchers {
  implicit val logger = Logger(Option(new TestAdapter()))

  val inputs = Seq(Node(0, Input()), Node(1, Input()),Node(2, Input()),Node(3, Input()),Node(4, Input()))
  val hidden = Seq(Node(5, Sigma()), Node(6, Sigma()),Node(7, Sigma()),Node(8, Sigma()), Node(9, Sigma()))
  val out = Seq(Node(-1, Sum()))
  val edgesInputHiddenOne: Seq[Edge] = (for (i <- 0 until 5) yield  for (j <- 0 until 5) yield Edge(inputs.apply(i), hidden.apply(j), Math.random())).flatten
  val edgesHiddenOneOut: Seq[Edge] = for (i <- 0 until 5) yield Edge(hidden.apply(i), out.apply(0), Math.random())
  val netTest = new Network(edgesInputHiddenOne ++ edgesHiddenOneOut, 1)


  def getNetwork: Network = {
    getPop(1).model.asInstanceOf[Network]
  }

  def getPop(hiddenLayers: Int): PopulationMember = {
    PopulationSearch.spawnNet(
      5,hiddenLayers, PopulationSearch.getRandomHiddenLayerVolume(5),
      PopulationSearch.fullWeave, Sigma())
  }

  def getMutator(params: MutationParameters, pop: PopulationMember): MutatingNetwork= {
    val network = pop.model.asInstanceOf[Network]
    val features = pop.features.asInstanceOf[NetworkFeatures]
    new MutatingNetwork(network.weights, network.lRate,  params, features)
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
    hiddenOne.nonEmpty should equal(true)
    hiddenTwo.nonEmpty should equal(true)
    hiddenOne.length should equal(hiddenTwo.length)
  }
  "Any spawned net" should "have a node with id -1, which is the output node" in {
    val net: Network = getNetwork
    net.weights.exists(_.to.id == -1) should equal(true)
    net.weights.exists(_.from.id == -1) should equal(false)
  }

  "incrementLayerIndexBy  " should "increase the index by X amount" in {
    val layers = getPop(1).features.asInstanceOf[NetworkFeatures].layers
    val increased = MutatingNetwork.incrementLayerIndexBy(layers, 1)
    val layerSum= layers.flatMap(_.map(_.layer.getOrElse(0))).sum
    val increasedSum = increased.flatMap(_.map(_.layer.getOrElse(0))).sum
    assert(layerSum + layers.length == increasedSum, "the layerindex did not increase by 1 in each layer")
  }

  "Adding a Layer " should "never result in disconnecting the outputnode, and add one layer" in {
    val pop = getPop(1)
    val params = MutationParameters(1.1, 0, 0, 0)
    val mutator = getMutator(params, pop)
    var mutated = mutator.mutate()
    var lastLayerCount = mutated.weights.map(_.to.layer).max.get
    for (i <- 0 until 100) {
      mutated = mutated.mutate()
      assert(mutated.weights.exists(_.to.id == -1), "no edge to output node left")
      val layerCount = mutated.weights.map(_.to.layer).max.get
      assert(layerCount > lastLayerCount, "did not increase layer-count")
      lastLayerCount = layerCount
    }
  }

  "Removing a Layer " should "never result in disconnecting the outputnode, and remove one layer" in {
    val pop = getPop(100)
    val params = MutationParameters(0, 1, 0, 0)
    val mutator = getMutator(params, pop)
    var mutated = mutator.mutate()
    var lastLayerCount = mutated.weights.map(_.to.layer).max.get
    for (i <- 0 until 99) {
      mutated = mutated.mutate()
      assert(mutated.weights.exists(_.to.id == -1), "no edge to output node left")
      val layerCount = mutated.weights.map(_.to.layer).max.get
      assert(layerCount < lastLayerCount, "did not decrease layer-count")
      lastLayerCount = layerCount
    }
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
