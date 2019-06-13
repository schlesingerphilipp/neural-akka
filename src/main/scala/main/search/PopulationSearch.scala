package main.search
import main.NeuralNet.{Edge, Function, Input, Model, Network, Node, Sigma, Sum}
import main.data.{Data, DataPoint, SplitData}
import main.search.PopulationSearch.interWeave
import main.util.{Logger, Logging}

import scala.annotation.tailrec
import scala.util.Random


class PopulationSearch(data: Data, popSize: Int, popNumber: Int) (implicit l: Logger) extends Logging {
  override val logger = l
  val split: (Seq[DataPoint], Seq[DataPoint]) = data.getTrainingTestSplit(0.7)
  val training = SplitData(split._1)
  val testSplit: (Seq[DataPoint], Seq[DataPoint]) = SplitData(split._2)getTrainingTestSplit 0.7
  val test = SplitData(testSplit._1)
  val validation = SplitData(testSplit._2)
  val inputs = data.data.apply(0).features.size
  val populations: Seq[Population] = for (i <- 0 until popNumber) yield PopulationSearch.createPopulation(inputs, popSize, i)

  def fit(): Model = {
    val firstStep = fitAndMutate(populations, training)
    val lastFitStep = fitRec(firstStep, training)

    lastFitStep.populations.maxBy(_.members.map(_.model.getMSE(data)).sum).members.maxBy(_.model.getMSE(data)).model
  }

  def trainModels(populations: Seq[Population], data: Data): Seq[Population] = {
    populations.map(p => p.members.map(m => (m.model.train(data, 0.01), m.features)))
      .map(seqOfSeq => Population(seqOfSeq.map(m => PopulationMember(m._1, m._2))))
  }
  def mutateModels(populations: Seq[Population], replicationFactor: Int): Seq[Population] = {
    populations.map(p=>Population(p.members.flatMap(_.mutateFrom(replicationFactor))))
  }

  def getScore(populations: Seq[Population], data: Data): (Double, Double)  = {
    val mses: Seq[Double] = populations.flatMap(_.members.map(_.model.getMSE(data)))
    (mses.sorted(Ordering[Double]).head, mses.sum / mses.length)
  }

  private case class FitStepResult(bestMse: Double, avrgMse: Double, populations: Seq[Population])

  private def fitAndMutate(populations: Seq[Population], data: Data): FitStepResult = {
    val mutated = mutateModels(populations, 1)//we do not have resizing jet
    val trained = trainModels(mutated, data)
    val scoring = getScore(trained, data)
    FitStepResult(scoring._1, scoring._2, mutated)
  }

  @tailrec
  private final def fitRec(step: FitStepResult, data: Data): FitStepResult = {
    val nextStep = fitAndMutate(step.populations, data)
    if (nextStep.bestMse / step.bestMse > 1.05 && nextStep.avrgMse / step.avrgMse > 1.05) {
      return nextStep
    }
    info("fitting step")
    info(s"Prediction: ${nextStep.populations.apply(0).members.apply(0).model.predict(data.data.apply(0).features)}")
    info(s"Target: ${data.data.apply(0).target}")
    fitRec(nextStep, data)

  }

  def predict(features: Seq[Seq[Double]]): Seq[Double] = ???

}
case class Population(members: Seq[PopulationMember])



//network edges -> delete 5% add 5%
case class PopulationMember(model: Model, features: ModelFeatures)(implicit l: Logger) {

  def mutateFrom(replicationFactor: Int): Seq[PopulationMember] = {
    for (i <- 0 until replicationFactor) yield mutate()
  }
  def mutate(): PopulationMember = {
    model match {
      case n: Network => {
        val params = MutationParameters(0.05, 0.05, 0.05, 0.05)
        val mutator = features match {
          case f: NetworkFeatures=> new MutatingNetwork(n.weights, n.lRate,  params, f)}
        val mutated = mutator.mutate()
        PopulationMember(mutated, mutated.features)
      }
    }
  }

}
trait ModelFeatures
case class NetworkFeatures(layers: Seq[Seq[Node]])(implicit l: Logger) extends ModelFeatures with Logging {
  override val logger = l
  val inputs: Int = layers.flatten.count(_.function match {
    case _: Input => true
    case _ => false
  })
}

case class MutationParameters(addLayerChance: Double, removeLayerChance: Double, addEdgeChance: Double, removeEdgeChance: Double)
object MutatingNetwork {
  def incrementLayerIndexBy(layersToIncrement: Seq[Seq[Node]], incrementBy: Int)(implicit l:Logger): Seq[Seq[Node]] = {
    layersToIncrement.map(_.map(n=>Node(n.id, n.function, n.layer.map(_ + incrementBy))))
  }
}
class MutatingNetwork(weights: Seq[Edge], lRate: Double, params: MutationParameters, val features: NetworkFeatures)
                     (implicit l:Logger) extends Network(weights, lRate)(l) {
  val layerCount: Int = features.layers.length

  def mutate(): MutatingNetwork = {
    try {
      return removeOneLayer().addOneLayer().addEdges().removeEdges()
    } catch {
      case e: Throwable => {
        error(e.getMessage)
        info("Exception with mutator")
        info(s"layers ${this.layerCount}")
        info(s"edges${this.weights.length}")
        info(s"nodes ${this.features.layers.flatten.length}")
        throw e
      }
    }
  }

  private def removeLayer(): MutatingNetwork = {
    //Do not remove input layer
    //Do not remove output layer
    //weave the new neighbours
    if (layerCount == 2) {
      info("Nothing to remove left")
      return this
    }
    val index = getRandomHiddenLayerIndex()
    val removedNodes = features.layers.apply(index).map(_.id)
    val remainingLayers: Seq[Seq[Node]] = features.layers.slice(0,index) ++
      MutatingNetwork.incrementLayerIndexBy(features.layers.slice(index +1, layerCount), -1)
    val toNodes = remainingLayers.apply(index)
    val fromNodes = remainingLayers.apply(index -1)
    val newEdges = interWeave(fromNodes, toNodes, PopulationSearch.fullWeave, () => Math.random())
    val remainingEdges = weights.filter(e=> !removedNodes.contains(e.from.id) || !removedNodes.contains(e.to.id))
    val updatedRemaining = updateLayerIndexOfRemainingEdges(remainingEdges, index, -1)
    val newWeights = updatedRemaining ++ newEdges
    new MutatingNetwork(newWeights, lRate, params, NetworkFeatures(remainingLayers))
  }

  private def addLayer(): MutatingNetwork = {
    //insert at random index
    //shift everything from the index 1 up
    val index = if (layerCount == 2) 1 else getRandomHiddenLayerIndex()
    debug(s"adding layer at ${index}")
    debug(s"in: ${features.layers.length} layered network")
    val nextId = features.layers.flatMap(_.map(_.id)).max + 1
    val newLayer = PopulationSearch.randomHiddenLayer(nextId, features.inputs, index)
    val newLayers = features.layers.slice(0, index) ++ Seq(newLayer) ++
      MutatingNetwork.incrementLayerIndexBy(features.layers.slice(index, layerCount), 1)
    // weave downwards and upwards
    val layerBelow = newLayers.apply(index - 1)
    val layerAbove = newLayers.apply(index + 1)
    val newEdgesTo = interWeave(layerBelow, newLayer, PopulationSearch.fullWeave, () => Math.random())
    val newEdgesFrom = interWeave(newLayer, layerAbove, PopulationSearch.fullWeave, () => Math.random())
    //Remove edges between old neighbours
    val belowIds = layerBelow.map(_.id)
    val aboveIds = layerAbove.map(_.id)
    val remainingEdges = weights.filter(e=> !(belowIds.contains(e.from.id) || aboveIds.contains(e.to.id)))
    val updatedRemaining = updateLayerIndexOfRemainingEdges(remainingEdges, index, 1)
    val newEdges = (updatedRemaining ++ newEdgesTo ++ newEdgesFrom).sortBy(_.from.layer)
    new MutatingNetwork(newEdges, lRate, params, NetworkFeatures(newLayers))
  }

  private def updateLayerIndexOfRemainingEdges(edges: Seq[Edge], editedIndex: Int, increase: Int): Seq[Edge] = {
    edges.map(e => {
      val from = MutatingNetwork.incrementLayerIndexBy(Seq(Seq(e.from)),
        if (e.from.layer.exists(_ >= editedIndex)) increase else 0).flatten.head
      val to = MutatingNetwork.incrementLayerIndexBy(Seq(Seq(e.to)),
        if (e.to.layer.exists(_ >= editedIndex)) increase else 0).flatten.head
      Edge(from, to, e.weight)
    })
  }

  /**
    * selects a random layer, but never the input or output layer
    * @return the index of one of the hidden layers
    */
  private def getRandomHiddenLayerIndex(): Int = {
    val candidate = Math.ceil(layerCount * Math.random()).toInt
    if (candidate >= layerCount -1) layerCount - 2 else candidate
  }

  private def addOneLayer(): MutatingNetwork = {
    if (Math.random() < params.addLayerChance ) {
      addLayer()
      } else this
  }
  private def removeOneLayer(): MutatingNetwork = {
    if (Math.random() < params.removeLayerChance ) {
      removeLayer()
    } else this
  }

  /**
    * For every non existing edge, add this edge with the chance of params.addEdgeChance
    * @return A Network with these added Edges
    */
  private def addEdges(): MutatingNetwork = {
    val newWeights = for (i <- 0 until features.layers.length - 2) yield newEdges(this, features.layers.apply(i),
      features.layers.apply(i+1), params.addEdgeChance)
    new MutatingNetwork(weights ++ newWeights.flatten, lRate, params, features)
  }

  /**
    * For every existing edge, remove this edge with the chance of params.removeEdgeChance,
    * iff all constraints to allow the removal are satisfied.
    * Constraints are:
    *   - Every Node must have at least one outgoing edge. Expect out node
    *   - Every Node must have at least one incoming edge. Expect Inputs
    * @return A Network without the removed Edges
    */
  private def removeEdges(): MutatingNetwork = {
    val newEdges = removeOneEdge(Random.shuffle(weights), Seq.empty)
    new MutatingNetwork(newEdges, lRate, params, features)
  }

  @tailrec
  private def removeOneEdge(left: Seq[Edge], right: Seq[Edge]): Seq[Edge] = {
    if (left.isEmpty) {
      return right
    }
    val remainIf = (Math.random() > params.removeEdgeChance
                  || left.count(_.from.id == left.head.from.id) == 1
                  || left.count(_.to.id == left.head.to.id) == 1)
    val remain = if (remainIf) Seq(left.head) else Seq.empty
    removeOneEdge(left.tail, right ++ remain)
  }

  private def newEdges(net: Network, from: Seq[Node], to: Seq[Node], chance: Double): Seq[Edge] = {

    val newEdges = from.flatMap(fromNode => to.flatMap(toNode => newEdgeIfNotExists(net, fromNode, toNode)))
      //.filter(_=>Math.random() < chance)
    newEdges
  }

  private def newEdgeIfNotExists(net: Network, from: Node, to: Node) : Option[Edge] = {
    if (!net.weights.exists(e=>e.from.id == from.id && e.to.id == to.id)) {
      return Option(Edge(from, to, Math.random()))
    }
    Option.empty
  }

}



object PopulationSearch {
  def createPopulation(inputs: Int, size: Int, id: Int)(implicit l: Logger): Population = {
    val members = for (i <- 0 until size) yield
      spawnNet(inputs, (Math.random() * inputs).toInt, getRandomHiddenLayerVolume(inputs), this.fullWeave, Sum())
    Population(members)
  }

  def randomHiddenLayer(nodeIdSeq: Int, inputs: Int, hiddenLayerIndex: Int)(implicit l: Logger): Seq[Node] = {
    val numOfNodes = Math.ceil(2 * Math.random() * inputs).toInt
    for (i <- 0 until numOfNodes) yield Node(i + nodeIdSeq, Sigma(), Option(hiddenLayerIndex))
  }

  def spawnNet(inputs: Int, hiddenLayers: Int, hiddenLayerVolume: (Int, Int, Int)=>Int,
               interWeaveFunc: (Node, Seq[Node]) => Seq[(Node,Node)], output: Function)(implicit l: Logger)
  : PopulationMember = {
    val inputNodes = {
      for (i <- 0 until inputs) yield Node(i, Input(), Option(0))
    }
    def hiddenLayer(hiddenLayerIndex: Int, idSequence: Int): Seq[Node] = {
      for (i <- 0 until hiddenLayerVolume(inputs,hiddenLayers, hiddenLayerIndex)) yield Node(i + idSequence, Sigma(),
        Option(hiddenLayerIndex))
    }
    @tailrec
    def makeHiddenLayers(hiddenLayers: Int, index: Int, idSeq: Int, acc: Seq[Seq[Node]]): Seq[Seq[Node]] = {
      if (hiddenLayers < index) {
        return acc
      }
      val layer = hiddenLayer(index, idSeq + 1)
      makeHiddenLayers(hiddenLayers, index +1, layer.reverse.head.id , acc :+ layer)
    }
    val hiddenLayersList = makeHiddenLayers(hiddenLayers, 1, inputs +1, Seq.empty)
    val out = Seq(Node(-1, output, Option(hiddenLayers + 1)))
    val layers: Seq[Seq[Node]] = inputNodes +: hiddenLayersList :+ out
    val weights = layers.flatMap(layer => {
      if (layers.indexOf(layer) <= layers.length -2) {
          interWeave(layer, layers.apply(layers.indexOf(layer) +1) , this.fullWeave, () => Math.random())
      } else {
        Seq.empty
      }
    })
    PopulationMember(new Network(weights, 0.01), NetworkFeatures(layers))
  }
  def interWeave(from: Seq[Node], to: Seq[Node], interWeaveFunc: (Node, Seq[Node]) => Seq[(Node,Node)],
                 weightFunction: ()=>Double): Seq[Edge] = {
      from.flatMap(interWeaveFunc(_, to)).map(tuple=>Edge(tuple._1, tuple._2, weightFunction()))
  }
  def fullWeave(from: Node, nextLayer: Seq[Node]): Seq[(Node,Node)] = {
      nextLayer.map((from, _))
  }
  def partialWeave(from: Node, nextLayer: Seq[Node]): Seq[(Node,Node)] = {
    val density = 0.5
    fullWeave(from,nextLayer).filter(_=>Math.random() > density)
  }
  def getRandomHiddenLayerVolume(inputs: Int): (Int, Int, Int) => Int = {
    (a: Int, b: Int, c: Int) => Math.ceil(2 * Math.random() * inputs).toInt
  }
}

