package main.neuralnet.domain

import jdk.jshell.spi.ExecutionControl.NotImplementedException
import main.neuralnet.domain

import scala.annotation.tailrec


case class NextLayeredNetwork(layerss: Seq[Layer])
  extends AbstractLayeredNetwork(layerss) {


  def getDefinitionFromInstance(model: LayeredNetwork): NextModelParameters = {
    val inputLayerDefinition: Seq[InputNodeDefinition] = layers.head.nodes
      .map(node => InputNodeDefinition(node.function, node.asInstanceOf[InputNode].inputId))
    val hiddenLayers: Seq[Layer] = layers.tail.slice(0, layers.tail.length -1)
    val hiddenLayerDefinitions: Seq[Seq[HiddenNodeDefinition]] =
      hiddenLayers.map(layer => layer.nodes.map(node => HiddenNodeDefinition(node.function)))
    NextModelParameters(inputLayerDefinition, hiddenLayerDefinitions)
  }

  def predict(dp: NextDataPoint): Double = {
    val nodeValuesAndInputs: (Map[Int, Double], Map[Int, Seq[Value]]) = getAllNodeValues(dp)
    // Node with id == -1 is the output node
    nodeValuesAndInputs._1.apply(-1)
  }

  def getMSE(data: Seq[NextDataPoint]): Double = {
    val predictions: Seq[(Double, Double)] = data.map(d => (predict(d), d.target))
    val squareError = predictions.map(p => Math.pow(p._1 - p._2, 2)).sum
    squareError / data.length
  }

  def train(data: Seq[NextDataPoint]): NextLayeredNetwork = {
    trainStep(data.head, data.tail)
  }

  @tailrec
  final def trainStep(dataPoint: NextDataPoint, otherData: Seq[NextDataPoint]): NextLayeredNetwork = {
    val nextNetwork = backPropagation(dataPoint, 0.05)
    if (otherData.isEmpty) {
      return nextNetwork
    }
    nextNetwork.trainStep(otherData.head, otherData.tail)
  }

  def updateNodes(layer: Layer, nodeInputs: Map[Int, Seq[Value]], nodeDeltas: Map[Int, Double],
                  nodeValues: Map[Int, Double]): Layer = {
    val nodes = updateNodeFunctions(layer, nodeInputs, nodeDeltas, nodeValues)
    layer match {
      case _: InputLayer => InputLayer(nodes.map {
        case in: InputNode => in
      })
      case _: SimpleLayer => SimpleLayer(nodes, layer.inputWeights)
    }
  }


  def updateNodeFunctions(layer: Layer, nodeInputs: Map[Int, Seq[Value]], nodeDeltas: Map[Int, Double],
                                nodeValues: Map[Int, Double]): Seq[ValuableNode] = {
    layer.nodes.map(n => {
      val updatedFunction: Function = n.function match {
        case func: NextFunction =>
          NextFunction(func.model.train(nodeInputs.apply(n.id), nodeValues.apply(n.id) - nodeDeltas.apply(n.id)))
        case f: Function => f
      }
      n match {
        case i: InputNode => InputNode(updatedFunction, i.id, i.inputId)
        case v: ValueNode => ValueNode(updatedFunction, v.id)
      }
    })
  }

  protected final def backPropagation(input: NextDataPoint, lRate: Double): NextLayeredNetwork = {
    val nodeValuesAndInputs = getAllNodeValues(input)
    val nodeValues = nodeValuesAndInputs._1
    val nodeInputs = nodeValuesAndInputs._2
    val nodeDeltas: Map[Int, Double] = getAllDeltaE(Option.empty, layers.reverse.head, layers.reverse.tail, nodeValues,
      input.target, Map.empty)
    val updatedWeights = layers.map(updateLayer(_, nodeDeltas, nodeValues, input.target, lRate))
    NextLayeredNetwork(updatedWeights.map(updateNodes(_, nodeInputs, nodeValues, nodeDeltas)))
  }

  override def applyWeightsUpdate(layer: Layer, newWeights: Map[Edge, Double]): Layer = {
    layer match {
      case in: InputLayer => in
      case _ => SimpleLayer(layer.nodes, newWeights)
    }
  }

  /**
    * Feed forwards the input through the network and captures the node values.
    *
    * @param input feature list
    * @return Map [nodeId -> nodeValue] of all nodes in the network
    */
  def getAllNodeValues(dp: NextDataPoint): (Map[Int, Double], Map[Int, Seq[Value]]) = {
    val nextLayerIns = mapInputValues(layers.head, dp)
    val nextLayerValues = mapWeightsAndValues(nextLayerIns, layers.tail.head)
    getAllNodeValuesStep(layers.head, layers.tail, nextLayerValues, Map.empty)
  }

  def mapInputValues(inputLayer: Layer, dp: NextDataPoint): Map[Edge, Double] = {
    val ins: Map[Int, Double] = inputLayer match {
      case in: InputLayer => in.inputNodes.map(n=> n.id -> n.getOut(dp)).toMap
      case _ => throw new NotImplementedException("Unexpected match clause reached")
    }
    mapLayerOutsToNextLayer(ins, layers.tail.head)
  }
}



case class InputNode(func: Function, idd: Int, inputId: String) extends ValuableNode(func, idd) {
  def getOut(dp: NextDataPoint): Double = {
    function.getValue(dp.features.apply(inputId).map(Value(1, _)))
  }
}
case class NextFunction(model: ExternalModel) extends Logarithmic {

  override def getValue(input: Seq[Value]): Double = {
    val z = model.predict(input.map(i=> i.weight * i.feature))
    DoublePrecision.sigma(1 / (1+ Math.pow(Math.E, -1 * z)))
  }
}
case class InputLayer(inputNodes: Seq[InputNode]) extends AbstractLayer(inputNodes, Map.empty)
case class NextDataPoint(target: Double, features: Map[String, Seq[Double]])
case class ExternalModel() {
  def predict(input: Seq[Double]): Double = ???
  def train(input: Seq[Value], target: Double): ExternalModel = ???
}