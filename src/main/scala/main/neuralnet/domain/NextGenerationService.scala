package main.neuralnet.domain

import scala.annotation.tailrec


class NextGenerationService() extends GenerationService {


  def generateDefinition(definition: NextModelParametersDefinition): NextModelParameters = {
    val hiddenLayers = for (i <- 0 until definition.hiddenLayers)
      yield generateHiddenLayerDefinition(definition.minHiddenLayerVolume, definition.maxHiddenLayerVolume,
        definition.hiddenFunctionSelector, definition.hiddenFunctionParameters)
    val inputLayer = generateInputLayerDefinition(definition.inputs, definition.inputFunctionSelector,
      definition.inputFunctionParameters)
    NextModelParameters(inputLayer, hiddenLayers)
  }

  def generateHiddenLayerDefinition(minHiddenLayerVolume: Int, maxHiddenLayerVolume: Int,
                                    hiddenFunctionSelector: SelectorParameters => Function,
                                    hiddenFunctionParameters: SelectorParameters): Seq[HiddenNodeDefinition] = {
    val nodeCount = nextInt(minHiddenLayerVolume, maxHiddenLayerVolume)
    for (i <- 0 until nodeCount) yield HiddenNodeDefinition(hiddenFunctionSelector(hiddenFunctionParameters))
  }

  def generateInputLayerDefinition(inputs: Seq[String], inputFunctionSelector: SelectorParameters => Function,
                                   inputFunctionParameters: SelectorParameters): Seq[InputNodeDefinition] = {
    inputs.map(input => InputNodeDefinition(inputFunctionSelector(inputFunctionParameters), input))
  }

  def generateModel(modelParameters: NextModelParameters): NextLayeredNetwork = {
    val inputLayer = generateInputLayer(modelParameters.inputLayerDefinition)
    val idCounter = inputLayer.nodes.map(n=>n.id).max
    val inputAndHiddenLayers = nextGenerateHiddenLayers(modelParameters.hiddenLayerDefinitions, idCounter, Seq(inputLayer))
    val outputLayer = generateOutputLayer(inputAndHiddenLayers.reverse.head)
    val layers: Seq[Layer] = inputAndHiddenLayers :+ outputLayer
    NextLayeredNetwork(layers)
  }

  @tailrec
  final def nextGenerateHiddenLayers(layerDefinitions: Seq[Seq[HiddenNodeDefinition]], idCounter: Int, accumulator: Seq[Layer]): Seq[Layer] = {
    if (layerDefinitions.isEmpty) {
      return accumulator
    }
    val layer = generateHiddenLayer(accumulator.reverse.head, layerDefinitions.head, idCounter)
    val nextId = layer.nodes.length + idCounter + 1
    nextGenerateHiddenLayers(layerDefinitions.tail, nextId, accumulator :+ layer)
  }

  def generateHiddenLayer(previousLayer: Layer, nodeDefinitions: Seq[HiddenNodeDefinition], idCounter: Int): Layer = {
    val nodes: Seq[ValueNode] = generateHiddenNodes(nodeDefinitions, idCounter)
    val inputWeights: Map[Edge, Double] = generateInputWeights(nodes, previousLayer)
    SimpleLayer(nodes, inputWeights)
  }


  def generateInputLayer(nodeDefinitions: Seq[InputNodeDefinition]): InputLayer = {
    val nodes = nodeDefinitions.zipWithIndex.map(e=> new InputNode(e._1.function, e._2, e._1.dataSetId))
    InputLayer(nodes)
  }


  def generateHiddenNodes(nodes: Seq[HiddenNodeDefinition], idCounter: Int): Seq[ValueNode] = {
    nodes.zipWithIndex.map(e=> ValueNode(e._1.function, e._2 + idCounter))
  }



}
case class NextModelParametersDefinition(inputs: Seq[String],
                                         inputFunctionParameters: SelectorParameters,
                                         inputFunctionSelector: (SelectorParameters) => Function,
                                         hiddenFunctionParameters: SelectorParameters,
                                         hiddenFunctionSelector: SelectorParameters => Function,
                                         hiddenLayers: Int,
                                         minHiddenLayerVolume: Int, maxHiddenLayerVolume: Int)
case class NextModelParameters(inputLayerDefinition: Seq[InputNodeDefinition],
                           hiddenLayerDefinitions: Seq[Seq[HiddenNodeDefinition]])
case class InputNodeDefinition(function: Function, dataSetId: String)
case class HiddenNodeDefinition(function: Function)

case class SelectorParameters()