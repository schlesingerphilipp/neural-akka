package main.search

import akka.event.LoggingAdapter
import main.NeuralNet.{Edge, Function, Input, Model, Network, Node, Sigma, Sum}
import main.data.{Data, DataPoint, SplitData}
import main.search.PopulationSearch.interWeave
import main.util.{Logger, Logging}

import scala.annotation.tailrec


class PopulationSearch(data: Data, popSize: Int, popNumber: Int) (implicit l: Logger) extends Logging {
  override val logger = l
  val split: (Seq[DataPoint], Seq[DataPoint]) = data.getTrainingTestSplit(0.7)
  val training = SplitData(split._1)
  val testSplit: (Seq[DataPoint], Seq[DataPoint]) = SplitData(split._2)getTrainingTestSplit 0.7
  val test = SplitData(testSplit._1)
  val validation = SplitData(testSplit._2)
  val inputs = data.data.apply(0).features.size
  val populations: Seq[Population] = for (i <- 0 until popSize) yield PopulationSearch.createPopulation(inputs, popSize, i)

  def fit(): Model = {
    val firstStep = fitAndMutate(populations, training)
    val lastFitStep = fitRec(firstStep, training)
    //loggerr.get.info("fitted")
    lastFitStep.populations.maxBy(_.members.map(_.model.getMSE(data)).sum).members.maxBy(_.model.getMSE(data)).model
  }

  def trainModels(populations: Seq[Population], data: Data): Seq[Population] = {
    populations.map(p => p.members.map(m => (m.model.train(data, 0.01), m.features)))
      .map(seqOfSeq => Population(seqOfSeq.map(m => PopulationMember(m._1, m._2))))
  }
  def mutateModels(populations: Seq[Population], replicationFactor: Int): Seq[Population] = {
    populations.flatMap(_.members.map(_.mutateFrom(replicationFactor))).map(Population)
  }

  def getScore(populations: Seq[Population], data: Data): (Double, Double)  = {
    val mses: Seq[Double] = populations.flatMap(_.members.map(_.model.getMSE(data)))
    (mses.sorted(Ordering[Double].reverse).head, mses.sum / mses.length)
  }

  private case class FitStepResult(bestMse: Double, avrgMse: Double, populations: Seq[Population])

  private def fitAndMutate(populations: Seq[Population], data: Data): FitStepResult = {
    val trained = trainModels(populations, data)
    val mutated = mutateModels(trained, 1)//we do not have resizing jet
    val scoring = getScore(mutated, data)
    FitStepResult(scoring._1, scoring._2, mutated)
  }

  @tailrec
  private final def fitRec(step: FitStepResult, data: Data): FitStepResult = {
    val nextStep = fitAndMutate(step.populations, data)
    if (nextStep.bestMse / step.bestMse < 1.05 && nextStep.avrgMse / step.avrgMse < 1.05) {
      return nextStep
    }
    info("fitting step")
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
        val params = MutationParameters(0.05, 0.05, 0.05)
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

private case class MutationParameters(mutateLayersChance: Double, addEdgeChance: Double, removeEdgeChance: Double)

class MutatingNetwork(weights: Seq[Edge], lRate: Double, params: MutationParameters, val features: NetworkFeatures)
                     (implicit l:Logger) extends Network(weights, lRate)(l) {
  val layerCount: Int = features.layers.length

  def mutate(): MutatingNetwork = {
    try {
      addOrRemoveOneLayer().addEdges().removeEdges()
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
    if (layerCount == 2) return this
    val index = getRandomHiddenLayerIndex()
    if (index == layerCount- 1) {
      info("this should not happen")
    }
    val removedNodes = features.layers.apply(index).map(_.id)
    val remainingLayers: Seq[Seq[Node]] = features.layers.slice(0,index) ++
      incrementLayerIndexBy(features.layers.slice(index +1, layerCount), -1)
    val toNodes = features.layers.apply(index + 1)
    val fromNodes = features.layers.apply(index - 1)
    val newEdges = interWeave(fromNodes, toNodes, (fromNodes.length + toNodes.length) / 2, PopulationSearch.fullWeave, () => Math.random())
    val remainingEdges = weights.filter(e=> !removedNodes.contains(e.from.id) || !removedNodes.contains(e.to.id))
    new MutatingNetwork(remainingEdges ++ newEdges, lRate, params, NetworkFeatures(remainingLayers))
  }

  private def incrementLayerIndexBy(layersToIncrement: Seq[Seq[Node]], incrementBy: Int): Seq[Seq[Node]] = {
    layersToIncrement.map(_.map(n=>Node(n.id, n.function, n.layer.map(_ + incrementBy))))
  }

  private def addLayer(): MutatingNetwork = {
    //insert at random index
    val index = if (layerCount == 2) 1 else getRandomHiddenLayerIndex()
    val maxId = features.layers.flatMap(_.map(_.id)).max
    val newLayer = PopulationSearch.randomHiddenLayer(maxId, features.inputs, index)
    val newLayers = features.layers.slice(0, index) ++ Seq(newLayer) ++
      incrementLayerIndexBy(features.layers.slice(index +1, layerCount), 1)
    // weave downwards and upwards
    val layerBelow = features.layers.apply(index-1)
    val layerAbove = features.layers.apply(index)
    val newEdgesTo = interWeave(layerBelow, newLayer, (layerBelow.length + newLayer.length) / 2, PopulationSearch.fullWeave, () => Math.random())
    val newEdgesFrom = interWeave(newLayer, layerAbove, (newLayer.length + layerAbove.length) / 2, PopulationSearch.fullWeave, () => Math.random())
    //Remove edges between old neighbours
    val belowIds = layerBelow.map(_.id)
    val aboveIds = layerAbove.map(_.id)
    val remainingEdges = weights.filter(e=> !(belowIds.contains(e.from.id) || aboveIds.contains(e.to.id)))
    val newEdges = remainingEdges ++ newEdgesTo ++ newEdgesFrom
    new MutatingNetwork(newEdges, lRate, params, NetworkFeatures(newLayers))
  }

  /**
    * selects a random layer, but never the input or output layer
    * @return the index of one of the hidden layers
    */
  private def getRandomHiddenLayerIndex(): Int = {
    val candidate = Math.ceil(layerCount * Math.random()).toInt
    if (candidate >= layerCount -1) layerCount - 2 else candidate
  }

  private def addOrRemoveOneLayer(): MutatingNetwork = {
    if (Math.random() > params.mutateLayersChance ) {
      if (Math.random() > 0.5) {
        addLayer()
      } else {
        removeLayer()
      }
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
    *   - Every Node must have at least one outgoing edge
    * @return A Network without the removed Edges
    */
  private def removeEdges(): MutatingNetwork = {
    val newEdges = weights.dropWhile(e=>Math.random() > 1 - params.removeEdgeChance && weights.count(_.from.id == e.from.id) > 1)
    new MutatingNetwork(newEdges, lRate, params, features)
  }

  private def newEdges(net: Network, from: Seq[Node], to: Seq[Node], chance: Double): Seq[Edge] = {
    from.flatMap(fromNode => to.flatMap(toNode => newEdgeIfNotExists(net, fromNode, toNode)))
      .filter(_=>Math.random() < 1 - chance)
  }

  private def newEdgeIfNotExists(net: Network, from: Node, to: Node) : Option[Edge] = {
    if (!net.weights.exists(e=>e.from.id == from.id && e.to.id == to.id)) {
      Option(Edge(from, to, Math.random()))
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
    val numOfNodes = (2 * Math.random() * inputs).toInt
    for (i <- 0 until numOfNodes) yield Node(i + nodeIdSeq, Sigma(), Option(hiddenLayerIndex))
  }

  def spawnNet(inputs: Int, hiddenLayers: Int, hiddenLayerVolume: (Int, Int, Int)=>Int,
               interWeaveFunc: (Int, Int, Node, Seq[Node]) => Seq[(Node,Node)], output: Function)(implicit l: Logger)
  : PopulationMember = {
    val interWeaveFactor = inputs
    val inputNodes = {
      for (i <- 0 until inputs) yield Node(i, Input(), Option(0))
    }
    def hiddenLayer(hiddenLayerIndex: Int, idSequence: Int): Seq[Node] = {
      for (i <- 0 until hiddenLayerVolume(inputs,hiddenLayers, hiddenLayerIndex)) yield Node(i + idSequence, Sigma(),
        Option(hiddenLayerIndex))
    }
    val hiddenLayersList: Seq[Seq[Node]] = {
      for (i <- 1 until hiddenLayers + 1) yield hiddenLayer(i, inputs + (i-1) * hiddenLayerVolume(inputs,hiddenLayers, i))
    }
    val out = Seq(Node(-1, output, Option(hiddenLayers + 1)))
    val layers: Seq[Seq[Node]] = inputNodes +: hiddenLayersList :+ out
    val weights = layers.flatMap(layer => {
      if (layers.indexOf(layer) <= layers.length -2) {
          interWeave(layer, layers.apply(layers.indexOf(layer) +1), interWeaveFactor , this.fullWeave, () => Math.random())
      } else {
        Seq.empty
      }
    })
    PopulationMember(new Network(weights, 0.01), NetworkFeatures(layers))
  }
  def interWeave(from: Seq[Node], to: Seq[Node], interWeaveFactor: Int,
                 interWeaveFunc: (Int, Int, Node, Seq[Node]) => Seq[(Node,Node)], weightFunction: ()=>Double): Seq[Edge] = {
      val actualWeaveFactor = if (interWeaveFactor < to.length) interWeaveFactor else to.length
      from.flatMap(f => {
      val idx: Int = from.indexOf(f)
      val startIndx = if (idx - actualWeaveFactor <  0) 0 else  idx - actualWeaveFactor
      val endIndx = if (idx + actualWeaveFactor >  to.length) to.length else  idx + actualWeaveFactor
      val nodeTuples = interWeaveFunc(startIndx, endIndx, f, to)
      nodeTuples.map(tuple=>Edge(tuple._1, tuple._2, weightFunction()))
    })
  }
  def fullWeave(startIndx: Int, endIndx: Int, from: Node, nextLayer: Seq[Node]): Seq[(Node,Node)] = {
      for (i <- startIndx until endIndx) yield (from, nextLayer.apply(i))
  }
  def partialWeave(startIndx: Int, endIndx: Int, from: Node, nextLayer: Seq[Node]): Seq[(Node,Node)] = {
    val density = 0.5
    fullWeave(startIndx,endIndx,from,nextLayer).filter(_=>Math.random() > density)
  }
  def getRandomHiddenLayerVolume(inputs: Int): (Int, Int, Int) => Int = {
    (a: Int, b: Int, c: Int) => (2 * Math.random() * inputs).toInt
  }
}

