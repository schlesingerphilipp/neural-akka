package main.search
import akka.event.LoggingAdapter
import main.NeuralNet.{Edge, Function, Input, Network, Node, Sigma, Sum}
import main.data.{Data, SplitData}
import main.search.Population

import scala.collection.immutable
class PopulationSearch(data: Data) {
  val split = data.getTrainingTestSplit(0.7)
  val training = SplitData(split._1)
  val testSplit = SplitData(split._2)getTrainingTestSplit(0.7)
  val test = SplitData(testSplit._1)
  val validation = SplitData(testSplit._2)
  def fitnessTest(individual: Network): Double = individual.getMSE(test)
  def breed(pop1: Population, pop2: Population): (Population, Population) =  {
    (pop1, pop2)
  }

}

class Population(members: Seq[Network]) {}


object PopulationSearch {
  def createPopulation(inputs: Int, size: Int)(implicit loggerr: Option[LoggingAdapter]): Population = {
    val members = for (i <- 0 until size +1) yield spawnFullSymetricRandom(inputs, (Math.random() * inputs).toInt, this.constantHiddenLayerVolume, Sum())
    new Population(members)
  }


  def spawnFullSymetricRandom(inputs: Int, hiddenLayers: Int, hiddenLayerVolume: (Int, Int, Int)=>Int, output: Function)(implicit loggerr: Option[LoggingAdapter]) :Network = {
    val interWeaveFactor = inputs
    val inputNodes = {
      for (i <- 0 until inputs) yield Node(i, Input())
    }
    val out = Seq(Node(-1, output))
    def hiddenLayer(hiddenLayerIndex: Int, idSequence: Int): Seq[Node] = {
      for (i <- 0 until hiddenLayerVolume(inputs,hiddenLayers, hiddenLayerIndex)) yield Node(i + idSequence, Sigma())
    }
    val hiddenLayersList: Seq[Seq[Node]] = {
      for (i <- 0 until hiddenLayers) yield hiddenLayer(i, inputs + i * hiddenLayerVolume(inputs,hiddenLayers, i))
    }
    val layers: Seq[Seq[Node]] = inputNodes +: hiddenLayersList :+ out
    val weights = layers.map(layer => {
      if (layers.indexOf(layer) <= layers.length -2) {
          interWeave(layer, layers.apply(layers.indexOf(layer) +1), interWeaveFactor, this.symetricWeave, () => Math.random())
      } else {
        Seq.empty
      }
    }).flatten
    new Network(weights, 0.01)
  }
  def interWeave(from: Seq[Node], to: Seq[Node], interWeaveFactor: Int, interWeaveFunc: (Int, Int, Node, Seq[Node]) => Seq[(Node,Node)], weightFunction: ()=>Double): Seq[Edge] = {
    from.map(f => {
      val idx: Int = from.indexOf(f)
      val startIndx = if (idx - interWeaveFactor <  0) 0 else  idx - interWeaveFactor
      val endIndx = if (idx + interWeaveFactor >  to.length -1) to.length -1 else  idx + interWeaveFactor
      val nodeTuples = interWeaveFunc(startIndx, endIndx, f, to)
      nodeTuples.map(tuple=>Edge(tuple._1, tuple._2, weightFunction()))
    }).flatten
  }
  def symetricWeave(startIndx: Int, endIndx: Int, from: Node, nextLayer: Seq[Node]): Seq[(Node,Node)] = {
      for (i <- startIndx until endIndx) yield (from, nextLayer.apply(i))
  }
  def constantHiddenLayerVolume(inputs: Int, hiddenLayers: Int, hiddenlayerIndex: Int): Int = {
    inputs
  }
}

