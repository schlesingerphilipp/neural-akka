package main.search

import akka.event.LoggingAdapter
import main.NeuralNet
import main.NeuralNet.{Edge, Function, Input, Model, Network, Node, Sigma, Sum}
import main.data.{Data, DataPoint, SplitData}
import main.search.PopulationSearch.interWeave

import scala.annotation.tailrec
import scala.collection.immutable


class PopulationSearch(data: Data, popSize: Int, popNumber: Int)(implicit loggerr: Option[LoggingAdapter]) {
  val split: (Seq[DataPoint], Seq[DataPoint]) = data.getTrainingTestSplit(0.7)
  val training = SplitData(split._1)
  val testSplit: (Seq[DataPoint], Seq[DataPoint]) = SplitData(split._2)getTrainingTestSplit 0.7
  val test = SplitData(testSplit._1)
  val validation = SplitData(testSplit._2)
  val inputs = data.data.apply(0).features.size
  val populations: Seq[Population] = for (i <- 0 until popSize) yield PopulationSearch.createPopulation(inputs, popSize, i)

  def fit(): Unit = {
    val firstStep = fitAndMutate(populations, training)
    val fitted = fitRec(firstStep, training)
    loggerr.get.info("fitted")
  }

  def train: Population => Population = ???


  def trainModels(populations: Seq[Population], data: Data): Seq[Population] = {
    populations.map(p => p.members.map(m => (m.model.train(data, 0.01), m.features)))
      .map(seqOfSeq => Population(seqOfSeq.map(m => PopulationMember(m._1, m._2))))
  }
  def mutateModels(populations: Seq[Population], replicationFactor: Int): Seq[Population] = {
    populations.map(_.members.map(_.mutateFrom(replicationFactor))).flatten.map(Population)
  }
  def reduceMemberCountByPopulationAndFit(populations: Seq[Population], data: Data): Seq[Population] = {
    val popCount = populations.length
    val averagePopulationSize = populations.map(_.members.length).sum / popCount
    val sortedPops = populations.sortBy(_.members.map(_.model.getMSE(data)).sum)
    for (i <- 0 until sortedPops.length) yield removeMembersByIndex(sortedPops.apply(i), i, popCount, averagePopulationSize)
  }
  //Removes few for early indcies and many for later
  def removeMembersByIndex(pop: Population, index: Int, pops: Int, averagePopSize: Int): Population = {
    Population(pop.members.slice(0, ((pops - index)/ pops) * averagePopSize))
  }

  def getScore(populations: Seq[Population], data: Data): (Double, Double)  = {
    val mses: Seq[Double] = populations.flatMap(_.members.map(_.model.getMSE(data)))
    (mses.sorted(Ordering[Double].reverse).head, mses.sum / mses.length)
  }

  private case class FitStepResult(bestMse: Double, avrgMse: Double, populations: Seq[Population])

  def fitAndMutate(populations: Seq[Population], data: Data): FitStepResult = {
    val trained = trainModels(populations, data)
    val mutated = mutateModels(trained, 3)
    val maintainSize = reduceMemberCountByPopulationAndFit(mutated, data)
    val scoring = getScore(maintainSize, data)
    FitStepResult(scoring._1, scoring._2, maintainSize)
  }

  @tailrec
  final def fitRec(step: FitStepResult, data: Data): FitStepResult = {
    val nextStep = fitAndMutate(step.populations, data)
    if (nextStep.bestMse / step.bestMse < 1.05 && nextStep.avrgMse / step.avrgMse < 1.05) {
      return nextStep
    }
    loggerr.get.info("fitting step")
    fitRec(nextStep, data)
  }

  def predict(features: Seq[Seq[Double]]): Seq[Double] = ???

}
case class Population(members: Seq[PopulationMember])



//network edges -> delete 5% add 5%
case class PopulationMember(model: Model, features: ModelFeatures) {
  def mutateFrom(replicationFactor: Int): Seq[PopulationMember] = {
    for (i <- 0 until replicationFactor) yield mutate()
  }
  def mutate(): PopulationMember = {
    model match {
      case n: Network => {
        features match {
          case f: NetworkFeatures => {
            val chance = 0.05
            val newWeights = for (i <- 0 until f.layers.length - 1) yield newEdges(n, f.layers.apply(i), f.layers.apply(i+1), chance)
            PopulationMember(new Network(n.weights.dropWhile(_ =>Math.random() > 1 - chance) ++ newWeights.flatten, n.lRate)(n.logger), f)
          }
        }
      }
    }
  }

  def newEdges(net: Network, from: Seq[Node], to: Seq[Node], chance: Double): Seq[Edge] = {
    from.flatMap(fromNode => to.flatMap(toNode => newEdgeIfNotExists(net, fromNode, toNode)))
      .filter(_=>Math.random() < 1 - chance)
  }

  def newEdgeIfNotExists(net: Network, from: Node, to: Node) : Seq[Edge] = {
    if (!net.weights.exists(e=>e.from.id == from.id && e.to.id == to.id)) {
      Seq(Edge(from, to, Math.random()))
    }
    Seq.empty
  }

}
trait ModelFeatures
case class NetworkFeatures(layers: Seq[Seq[Node]]) extends ModelFeatures



object PopulationSearch {
  def createPopulation(inputs: Int, size: Int, id: Int)(implicit loggerr: Option[LoggingAdapter]): Population = {
    val members = for (i <- 0 until size) yield
      spawnNet(inputs, (Math.random() * inputs).toInt, getRandomHiddenLayerVolume(inputs), this.fullWeave, Sum())
    Population(members)
  }


  def spawnNet(inputs: Int, hiddenLayers: Int, hiddenLayerVolume: (Int, Int, Int)=>Int, interWeaveFunc: (Int, Int, Node, Seq[Node]) => Seq[(Node,Node)], output: Function)(implicit loggerr: Option[LoggingAdapter]) :PopulationMember = {
    val interWeaveFactor = inputs
    val inputNodes = {
      for (i <- 0 until inputs) yield Node(i, Input(), Option(0))
    }
    def hiddenLayer(hiddenLayerIndex: Int, idSequence: Int): Seq[Node] = {
      for (i <- 0 until hiddenLayerVolume(inputs,hiddenLayers, hiddenLayerIndex)) yield Node(i + idSequence, Sigma(), Option(hiddenLayerIndex))
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
  def interWeave(from: Seq[Node], to: Seq[Node], interWeaveFactor: Int, interWeaveFunc: (Int, Int, Node, Seq[Node]) => Seq[(Node,Node)], weightFunction: ()=>Double): Seq[Edge] = {
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
  def getRandomHiddenLayerVolume(inputs: Int): (Int, Int, Int) => Int = {
    (a: Int, b: Int, c: Int) => (Math.random() * inputs).toInt
  }
}

