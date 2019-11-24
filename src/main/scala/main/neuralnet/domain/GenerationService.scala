package main.neuralnet.domain

import scala.annotation.tailrec

trait GenerationService {
  def generateModel(modelParameters: ModelParameters): Model
}

object LayeredNetworkGenerationService extends GenerationService {
  val random = new scala.util.Random

  def generateInputWeights(nodes: Seq[ValueNode], previous: Layer): Map[Edge, Double] = {
    previous.nodes.flatMap(
      from => nodes.map(to => (Edge(from.id, to.id), Math.random()))).toMap
  }
  override def generateModel(modelParameters: ModelParameters): Model = {
    val layers = generateLayers(Seq.empty, modelParameters.hiddenLayers + 2, modelParameters.inputs,
      modelParameters.minLayerVolume, modelParameters.maxLayerVolume, 0)
    LayeredNetwork(layers)
  }


  @tailrec
  private def generateLayers(accumulator: Seq[Layer], remainingLayers: Int, inputs: Int,
                             minLayerVolume: Int, maxLayerVolume: Int, idCounter: Int): Seq[Layer] = {
    if (remainingLayers == 0) {
      return accumulator
    }
    val nodeCount = if (accumulator.isEmpty) inputs else
      minLayerVolume + random.nextInt(maxLayerVolume - minLayerVolume)
    val nextLayer = if (accumulator.isEmpty) generateInputLayer(inputs)
    else if (remainingLayers == 1) generateOutputLayer(accumulator.reverse.head)
      else generateHiddenLayer(accumulator.reverse.head, nodeCount, idCounter)
    generateLayers(accumulator :+ nextLayer, remainingLayers -1, inputs, minLayerVolume, maxLayerVolume, idCounter + nodeCount)
  }

  def generateHiddenLayer(previousLayer: Layer, nodeCount: Int, idCounter: Int): Layer = {
    val nodes: Seq[ValueNode] = generateValueNodes(nodeCount, idCounter, () => Sigma())
    val inputWeights: Map[Edge, Double] = LayeredNetworkGenerationService.generateInputWeights(nodes, previousLayer)
    Layer(nodes, inputWeights)
  }
    def generateInputLayer(inputs: Int): Layer = {
    val nodes: Seq[ValueNode] = generateValueNodes(inputs, 0, () => Input())
    val inputWeights: Map[Edge, Double] = nodes.map(n => (Edge(null, n.id), 1.0)).toMap
    Layer(nodes, inputWeights)
  }
  def generateOutputLayer(previousLayer: Layer): Layer = {
    val nodes: Seq[ValueNode] = Seq(ValueNode(Sum(), -1))
    val inputWeights: Map[Edge, Double] = previousLayer.nodes.map(n => (Edge(n.id, -1), Math.random())).toMap
    Layer(nodes, inputWeights)
  }

  def generateValueNodes(nodeCount: Int, idCounter: Int, funcGen: () => Function): Seq[ValueNode] = {
    for (i <- 0 until nodeCount) yield ValueNode(funcGen(), idCounter + i)
  }

}
case class ModelParameters(inputs: Int, hiddenLayers: Int, minLayerVolume: Int, maxLayerVolume: Int)
