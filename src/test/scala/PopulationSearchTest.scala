import main.data.ExampleData
import main.search._
import main.util.{Logger, TestAdapter}
import org.scalatest._

import scala.annotation.tailrec

object PopulationSearchTest extends FlatSpec {


}
/**
class PopulationSearchTest extends FlatSpec with Matchers {
  implicit val logger = Logger(Option(new TestAdapter()))

  val inputs = Seq(Node(0, Input()), Node(1, Input()),Node(2, Input()),Node(3, Input()),Node(4, Input()))
  val hidden = Seq(Node(5, Sigma()), Node(6, Sigma()),Node(7, Sigma()),Node(8, Sigma()), Node(9, Sigma()))
  val out = Seq(Node(-1, Sum()))
  val edgesInputHiddenOne: Seq[Edge_] = (for (i <- 0 until 5) yield  for (j <- 0 until 5) yield Edge_(inputs.apply(i), hidden.apply(j), Math.random())).flatten
  val edgesHiddenOneOut: Seq[Edge_] = for (i <- 0 until 5) yield Edge_(hidden.apply(i), out.apply(0), Math.random())
  val netTest = new Network(edgesInputHiddenOne ++ edgesHiddenOneOut, 1)


  def getNetwork: Network = {
    getPop(1).model.asInstanceOf[Network]
  }

  def getPop(hiddenLayers: Int): PopulationMember = {
    PopulationSearch.spawnNet(
      5,hiddenLayers, PopulationSearch.getRandomHiddenLayerVolume(5),
      PopulationSearch.fullWeave, Sigma())
  }

  def asMutatingNetwork(params: MutationParameters, pop: PopulationMember): MutatingNetwork= {
    val network = pop.model.asInstanceOf[Network]
    val features = pop.features.asInstanceOf[NetworkFeatures]
    new MutatingNetwork(network.weights, network.lRate,  params, features)
  }

  def testForCyclesWithParameters(params: MutationParameters, mutations: Int): Unit = {
    @tailrec
    def hasCycle(pathes: Seq[Seq[Int]], network: Network): Boolean = {
      for (i <- 1 until 10)
      if (pathes.length > Math.pow(10, i)) {
        logger.info(s"More than ${Math.pow(10, i)}")
      }
      val longerPathes: Seq[Seq[Seq[Int]]] = pathes.map(p=>{
        val nextSteps = network.weights.filter(e=> e.from.id == p.last).map(_.to.id)
        nextSteps.map(i => p :+ i)
      })
      val _hasCycle = pathes.exists(p=>p.size != p.toSet.size)
      if (_hasCycle) {
        return true
      }
      if (longerPathes.isEmpty) {
        return false
      }
      return hasCycle(longerPathes.flatten, network)
    }
    val pop = getPop(5)
    var mutated = asMutatingNetwork(params, pop).mutate()
    for (i <- 0 until mutations) {
      mutated = mutated.mutate()
      val inputs = mutated.weights.filter(_.from.function.isInstanceOf[Input])
      assert(!hasCycle(inputs.map(e => e.from.id).distinct.map(Seq(_)), mutated),
        s"${params.addLayerChance}, ${params.removeLayerChance}, " +
          s"${params.addEdgeChance}, ${params.removeEdgeChance}")
    }
  }

  "Cycles in the network" should "not exist with mutationparameters 0.0, 0.0, 0.0, 1" in {
    testForCyclesWithParameters(MutationParameters(0.0, 0.0, 0.0, 1), 10)
  }
  it should "not exist with mutationparameters 0.0, 0.0, 1, 0.0" in {
    testForCyclesWithParameters(MutationParameters(0.0, 0.0, 1, 0.0), 10)
  }
    it should "not exist with mutationparameters 0.0, 1, 0.0, 0.0" in {
      testForCyclesWithParameters(MutationParameters(0.0, 1, 0.0, 0.0), 10)
    }
    it   should "not exist with mutationparameters 1, 0.0, 0.0, 0.0" in {
      testForCyclesWithParameters(MutationParameters(1, 0.0, 0.0, 0.0), 10)
    }
    it   should "not exist with mutationparameters 0.5, 0.5, 0.0, 0.0" in {
      testForCyclesWithParameters(MutationParameters(0.5, 0.5, 0.0, 0.0), 10)
    }
    it     should "not exist with mutationparameters 0.5, 0.0, 0.5, 0.0" in {
      testForCyclesWithParameters(MutationParameters(0.5, 0.0, 0.5, 0.0), 10)
    }
    it      should "not exist with mutationparameters 0.0, 0.5, 0.5, 0.0" in {
      testForCyclesWithParameters(MutationParameters(0.0, 0.5, 0.5, 0.0), 10)
    }
    it        should "not exist with mutationparameters 0.5, 0.0, 0.0, 0.5" in {
      testForCyclesWithParameters(MutationParameters(0.5, 0.0, 0.0, 0.5), 10)
    }
    it          should "not exist with mutationparameters 0.0, 0.5, 0.0, 0.5" in {
      testForCyclesWithParameters(MutationParameters(0.0, 0.5, 0.0, 0.5), 10)
    }
    it            should "not exist with mutationparameters 0.0, 0.0, 0.5, 0.5" in {
      testForCyclesWithParameters(MutationParameters(0.0, 0.0, 0.5, 0.5), 10)
    }
    it              should "not exist with mutationparameters 0.0, 0.5, 0.5, 0.5" in {
      testForCyclesWithParameters(MutationParameters(0.0, 0.5, 0.5, 0.5), 10)
    }
    it                should "not exist with mutationparameters 0.5, 0.0, 0.5, 0.5" in {
      testForCyclesWithParameters(MutationParameters(0.5, 0.0, 0.5, 0.5), 10)
    }
    it                  should "not exist with mutationparameters 0.5, 0.5, 0.0, 0.5" in {
      testForCyclesWithParameters(MutationParameters(0.5, 0.5, 0.0, 0.5), 10)
    }
    it                    should "not exist with mutationparameters 0.5, 0.5, 0.5, 0.0" in {
      testForCyclesWithParameters(MutationParameters(0.5, 0.5, 0.5, 0.0), 10)
    }
    it should "not exist with mutationparameters 0.5, 0.5, 0.5, 0.5" in {
      testForCyclesWithParameters(MutationParameters(0.5, 0.5, 0.5, 0.5), 10)
    }

  "The fitting algorithm "should " not throw runtime exceptions and should yield a Model" in {
    val data = ExampleData(5,5,5000)
    val search = new PopulationSearch(data, 10, 1)
    val model = search.fit()
    assert(model.getMSE(data) > 0, "A zero mse, is problematic. maybe.") //hahah
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
  "A FULL spawn" should "be fully connected between each layer" in {
    val net: MutatingNetwork = asMutatingNetwork(MutationParameters(0,0,0,0), getPop(1))
    val counts = net.features.layers.map(_.length)
    @tailrec
    def fullyConnectedCount(layerCounts: Seq[Int], acc: Int): Int = {
      if (layerCounts.length < 2) {
        return acc
      }
      fullyConnectedCount(layerCounts.tail, acc + layerCounts.head * layerCounts.tail.head)
    }
    net.weights.size should equal(fullyConnectedCount(counts, 0))
  }

  "Any spawned net" should "have a node with id -1, which is the output node" in {
    val net: Network = getNetwork
    net.weights.exists(_.to.id == -1) should equal(true)
    net.weights.exists(_.from.id == -1) should equal(false)
  }

  "incrementLayerIndexBy  " should "increase the index by X amount" in {
    val layers = getPop(1).features.asInstanceOf[NetworkFeatures].layers
    val X = 3
    val increased = MutatingNetwork.incrementLayerIndexBy(layers, X)
    for (i <- 0 until increased.length -1) {
      for (node <- increased.apply(i)) {
        //the layer index starts at 0 like a list.
        assert(node.layer.getOrElse(-1) == i + X , "the node does not have the expected layer index")
      }
    }
  }

  "Adding a Layer " should "never result in disconnecting the outputnode, and add one layer" in {
    val pop = getPop(1)
    val params = MutationParameters(1.1, 0, 0, 0)
    val mutator = asMutatingNetwork(params, pop)
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
    val mutator = asMutatingNetwork(params, pop)
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

  "Adding an Edge " should "never result in disconnecting any other node" in {
    val pop = getPop(10)
    def edgeIn(edge: Edge_, weigths: Seq[Edge_]): Boolean =  {
      weigths.exists(e => e.from.id == edge.from.id && e.to.id == edge.to.id)
    }
    val params = MutationParameters(0, 0, 1, 0)
    val mutator = asMutatingNetwork(params, pop)
    var mutated = mutator.mutate()
    var lastEdges = mutated.weights.filter(_=>true)
    for (i <- 0 until 99) {
      mutated = mutated.mutate()
      for (e <- lastEdges) {
        assert(edgeIn(e, mutated.weights), "previously existing edge is gone")
      }
      lastEdges = mutated.weights.filter(_=>true)
    }
  }

  it should "never add an already existing edge" in {
    val pop = getPop(100)
    val params = MutationParameters(0, 0, 1, 0)
    val mutator = asMutatingNetwork(params, pop)
    val withLessEdges = new MutatingNetwork(mutator.weights.filter(_=>Math.random() > 0.5), mutator.lRate, params, mutator.features)
    val lastEdges = withLessEdges.weights
    val mutated = withLessEdges.mutate()
    val newEdges = mutated.weights.filter(edge => !lastEdges.exists(e => e.from.id == edge.from.id && e.to.id == edge.to.id))
    assert(lastEdges.length + newEdges.length  == mutated.weights.length, "There was an unexpected number of new edges")
  }

  it should "add some edges" in {
    val pop = getPop(100)
    val params = MutationParameters(0, 0, 1, 0)
    val mutator = asMutatingNetwork(params, pop)
    val withLessEdges = new MutatingNetwork(mutator.weights.filter(_=>Math.random() > 0.5), mutator.lRate, params, mutator.features)
    val lastEdges = withLessEdges.weights
    val mutated = withLessEdges.mutate()
    assert(lastEdges.length < mutated.weights.length, "After adding edges, there should be more edges than before")
  }

  "Removing an Edge to the output node " should "not be possible" in {
    val pop = getPop(100)
    val params = MutationParameters(0, 0, 0, 1)
    val mutator = asMutatingNetwork(params, pop)
    def edgesToOut(net: Network): Int = {
      net.weights.count(_.to.id == -1)
    }
    val mutated = mutator.mutate()
    assert(edgesToOut(mutated) == edgesToOut(mutator), "Unexpected number of edges to out")
  }

  //TODO: Check if nodes in layers equal nodes in edges

  "Removing as many edges as possible " should "not remove a single node" in {
    val pop = getPop(100)
    val params = MutationParameters(0, 0, 0, 1)
    val mutator = asMutatingNetwork(params, pop)
    val mutated = mutator.mutate()
    for (node <- mutator.features.layers.flatten) {
      assert(mutated.weights.exists(e => e.from.id == node.id || e.to.id == node.id), s"Node ${node} was not in mutated")
    }
  }

  it should "always leave each node with one outgoing edge" in {
    val pop = getPop(100)
    val params = MutationParameters(0, 0, 0, 1)
    val mutator = asMutatingNetwork(params, pop)
    val mutated = mutator.mutate()
    for (node <- mutator.features.layers.flatten) {
      assert(mutated.weights.exists(_.from.id == node.id || node.id == -1), s"Node ${node} has no out going edge")
    }
  }
  it should "always leave each node with at least one incoming edge" in {
    val pop = getPop(100)
    val params = MutationParameters(0, 0, 0, 1)
    val mutator = asMutatingNetwork(params, pop)
    val mutated = mutator.mutate()
    for (node <- mutator.features.layers.flatten) {
      assert(mutated.weights.exists(_.to.id == node.id || node.function.isInstanceOf[Input]), s"Node ${node} has no out going edge")
    }
  }

  "All layers of a spawned net" should "have at least one node" in {
    val pop = asMutatingNetwork(MutationParameters(0,0,0,0), getPop(100))
    val emptyLayers = pop.features.layers.count(_.length < 1)
    assert(emptyLayers == 0, s"There were ${emptyLayers} Empty layers in ${pop.features.layers.length} layers over all")
  }

  "After any mutation the network " should "have at least one node in each layer" in {
    val pop = asMutatingNetwork(MutationParameters(1,1,1,1), getPop(100))
    val mutated = pop.mutate()
    val emptyLayers = mutated.features.layers.count(_.length < 1)
    assert(emptyLayers == 0, s"There were ${emptyLayers} Empty layers in ${mutated.features.layers.length} layers over all")
  }

}
**/