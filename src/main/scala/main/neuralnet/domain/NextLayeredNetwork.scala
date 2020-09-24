package main.neuralnet.domain

import jdk.jshell.spi.ExecutionControl.NotImplementedException

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
    val nodeValues: Map[Int, Double] = getAllNodeValues(dp)
    // Node with id == -1 is the output node
    nodeValues.apply(-1)
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

  protected final def backPropagation(input: NextDataPoint, lRate: Double): NextLayeredNetwork = {
    val nodeValues = getAllNodeValues(input)
    val nodeDeltas: Map[Int, Double] = getAllDeltaE(Option.empty, layers.reverse.head, layers.reverse.tail, nodeValues,
      input.target, Map.empty)
    NextLayeredNetwork(layers.map(updateLayer(_, nodeDeltas, nodeValues, input.target, lRate)))
  }

  override def selectLayerClass(layer: Layer, newWeights: Map[Edge, Double]): Layer = {
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
  def getAllNodeValues(dp: NextDataPoint): Map[Int, Double] = {
    val edges = mapInputValues(layers.head, dp)
    getAllNodeValuesStep(layers.head, layers.tail, edges, Map.empty)
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
}