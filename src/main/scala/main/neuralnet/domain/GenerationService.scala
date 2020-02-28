package main.neuralnet.domain

import scala.annotation.tailrec

object GenerationService {
  val random = new scala.util.Random
  def nextInt(low: Int, high: Int): Int = {
    low + random.nextInt(high)
  }
  def generateLayerDefinitionFrom(modelParameters: ModelParameters) = {
    val hiddenLayers = for (i <- 0 until modelParameters.hiddenLayers)
      yield nextInt(modelParameters.minLayerVolume, modelParameters.maxLayerVolume)
    val outputLayerNodes = 1
    Seq(modelParameters.inputs) ++ hiddenLayers :+ outputLayerNodes
  }

  def generateModel(modelParameters: ModelParameters): Model = {
    val layerDefinition = generateLayerDefinitionFrom(modelParameters)
    generateFromDefinition(layerDefinition)
  }

  def generateFromDefinition(layerSizeDefinition: Seq[Int]): LayeredNetwork = {
    LayeredNetwork(generateLayersFromDefinition(layerSizeDefinition, Seq.empty, 0))
  }


  @tailrec
  def generateLayersFromDefinition(layerDefinitions: Seq[Int], accumulator: Seq[Layer], idCounter: Int): Seq[Layer] = {
    if (layerDefinitions.isEmpty) {
      return accumulator
    }
    val nodeCount = layerDefinitions.head
    val nextLayer = if (accumulator.isEmpty) generateInputLayer(nodeCount)
    else if (layerDefinitions.length == 1) generateOutputLayer(accumulator.reverse.head)
    else generateHiddenLayer(accumulator.reverse.head, nodeCount, idCounter)
    generateLayersFromDefinition(layerDefinitions.tail, accumulator :+ nextLayer, idCounter + nodeCount)
  }

  def generateInputWeights(nodes: Seq[ValueNode], previous: Layer): Map[Edge, Double] = {
    previous.nodes.flatMap(
      from => nodes.map(to => (Edge(from.id, to.id), Math.random()))).toMap
  }

  def generateHiddenLayer(previousLayer: Layer, nodeCount: Int, idCounter: Int): Layer = {
    val nodes: Seq[ValueNode] = generateValueNodes(nodeCount, idCounter, Sigma())
    val inputWeights: Map[Edge, Double] = generateInputWeights(nodes, previousLayer)
    Layer(nodes, inputWeights)
  }
    def generateInputLayer(inputs: Int): Layer = {
    val nodes: Seq[ValueNode] = generateValueNodes(inputs, 0, Input())
    val inputWeights: Map[Edge, Double] = nodes.map(n => (Edge(null, n.id), 1.0)).toMap
    Layer(nodes, inputWeights)
  }
  def generateOutputLayer(previousLayer: Layer): Layer = {
    val nodes: Seq[ValueNode] = Seq(ValueNode(Sum(), -1))
    val inputWeights: Map[Edge, Double] = previousLayer.nodes.map(n => (Edge(n.id, -1), Math.random())).toMap
    Layer(nodes, inputWeights)
  }

  def generateValueNodes(nodeCount: Int, idCounter: Int, valueFunction: Function): Seq[ValueNode] = {
    for (i <- 0 until nodeCount) yield ValueNode(valueFunction, idCounter + i)
  }

}
case class ModelParameters(inputs: Int, hiddenLayers: Int, minLayerVolume: Int, maxLayerVolume: Int)
