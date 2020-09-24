package main.neuralnet.domain

import main.data.DataPoint

import scala.annotation.tailrec



case class LayeredNetwork(layerss: Seq[Layer]) extends AbstractLayeredNetwork(layerss) {

  def predict(input: Seq[Double]): Double = {
    val ins = mapInputToInputLayer(input, layers.head)
    val nodeValues: Map[Int, Double] = getAllNodeValues(input)
    // Node with id == -1 is the output node
    nodeValues.apply(-1)
  }

  def getMSE(data: Seq[DataPoint]): Double = {
    val predictions = data.map(d => (predict(d.features), d.target))
    val squareError = predictions.map(p=> Math.pow(p._1 - p._2, 2)).sum
    squareError / data.length
  }

  /**
    * Feed forwards the input through the network and captures the node values.
    * @param input feature list
    * @return Map [nodeId -> nodeValue] of all nodes in the network
    */
  def getAllNodeValues(input: Seq[Double]): Map[Int, Double] = {
    val ins: Map[Edge, Double] = mapInputToInputLayer(input, layers.head)
    getAllNodeValuesStep(layers.head, layers.tail, ins, Map.empty)
  }


  protected final def backPropagation(input: Seq[Double], target: Double, lRate: Double): LayeredNetwork = {
    val nodeValues = getAllNodeValues(input)
    val nodeDeltas: Map[Int, Double] = getAllDeltaE(Option.empty, layers.reverse.head, layers.reverse.tail, nodeValues,
      target, Map.empty)
    LayeredNetwork(layers.map(updateLayer(_, nodeDeltas, nodeValues, target, lRate)))
  }

  def train(data: Seq[DataPoint]): LayeredNetwork = {
    trainStep(data.head, data.tail)
  }

  @tailrec
  final def trainStep(dataPoint: DataPoint, otherData: Seq[DataPoint]): LayeredNetwork = {
    val nextNetwork = backPropagation(dataPoint.features, dataPoint.target, 0.05)
    if (otherData.isEmpty) {
      return nextNetwork
    }
    nextNetwork.trainStep(otherData.head, otherData.tail)
  }

}

trait Network {
  def selectLayerClass(layer: Layer, newWeights: Map[Edge, Double]): Layer = {
    SimpleLayer(layer.nodes, newWeights)
  }
}

abstract class AbstractLayeredNetwork(layerss: Seq[Layer]) extends Network{
  val layers = layerss

  @tailrec
  protected final def getAllNodeValuesStep(currentLayer: Layer, otherLayers: Seq[Layer], ins: Map[Edge, Double],
                                           accumulator: Map[Int, Double]): Map[Int, Double] = {
    val layerOuts: Map[Int,Double] = currentLayer.getOuts(ins)
    val nextAccumulator = accumulator ++ layerOuts
    if (otherLayers.isEmpty) {
      return nextAccumulator
    }
    val nextLayerIns = mapLayerOutsToNextLayer(layerOuts, otherLayers.head)
    getAllNodeValuesStep(otherLayers.head, otherLayers.tail, nextLayerIns, nextAccumulator)
  }


  protected def mapInputToInputLayer(input: Seq[Double], inputLayer: Layer): Map[Edge, Double] = {
    //The ids of the input nodes match the indices of the input Sequence
    inputLayer.inputWeights.keys.map(edge => edge -> input.apply(edge.to)).toMap
  }

  protected def mapLayerOutsToNextLayer(layerOuts: Map[Int,Double], nextLayer: Layer): Map[Edge, Double] = {
    nextLayer.inputWeights.keys.map(edge => edge -> layerOuts.apply(edge.from)).toMap
  }

  protected final def updateLayer(layer: Layer, nodeDeltas: Map[Int, Double], nodeValues : Map[Int, Double], target: Double,
                                  lRate: Double): Layer = {
    val newWeights = layer.inputWeights.map(edgeWeight => {
      val delta = nodeValues.apply(edgeWeight._1.from) * nodeDeltas.apply(edgeWeight._1.to)
      val newWeight = edgeWeight._2 + lRate * delta
      (Edge(edgeWeight._1.from, edgeWeight._1.to), newWeight)
    })
    selectLayerClass(layer, newWeights)
  }

  @tailrec
  protected final def getAllDeltaE(previous: Option[Layer], currentLayer: Layer, otherLayers: Seq[Layer], allValues: Map[Int, Double], target: Double,
                                   accumulator: Map[Int, Double]): Map[Int, Double] = {
    val nextAccumulator = currentLayer.nodes.map(getNodeDeltaE(_, currentLayer, previous,  accumulator, target, allValues))
      .toMap ++ accumulator
    if (otherLayers.isEmpty) {
      return nextAccumulator
    }
    getAllDeltaE(Option(currentLayer), otherLayers.head, otherLayers.tail, allValues, target, nextAccumulator)
  }

  protected def getNodeDeltaE(node: ValuableNode, currentLayer: Layer, nextLayer: Option[Layer], earlierDeltas: Map[Int, Double],
                              target: Double, allValues: Map[Int, Double]): (Int, Double) = {
    val edgesFromNode = nextLayer.map(_.inputWeights.filter(w => w._1.from == node.id)).getOrElse(Map.empty).keys.toSeq
    val toNodeDeltas: Seq[Double] = edgesFromNode.map(edge => earlierDeltas.apply(edge.to))
    (node.id, node.deltaE(allValues(node.id), target, toNodeDeltas))
  }

}
trait Layer {
  val nodes: Seq[ValuableNode]
  val inputWeights: Map[Edge, Double]
  def getOuts(ins: Map[Edge, Double]): Map[Int,Double]
  def mapInEdgesToValues(ins: Map[Edge, Double], id: Int): Seq[Value]
}
abstract class AbstractLayer(nodess: Seq[ValuableNode], inputWeightss: Map[Edge, Double]) extends Layer{
  val nodes = nodess
  val inputWeights: Map[Edge, Double] = inputWeightss
  def getOuts(ins: Map[Edge, Double]): Map[Int,Double] = {
    nodes.map(node => (node.id, node.getOut(inputWeights, ins))).toMap
  }

  def mapInEdgesToValues(ins: Map[Edge, Double], id: Int): Seq[Value] = {
    inputWeights.map(weight => Value(weight._2, ins.apply(weight._1))).toSeq
  }
}
case class SimpleLayer(nodess: Seq[ValuableNode], inputWeightss: Map[Edge, Double]) extends AbstractLayer(nodess, inputWeightss)

abstract class ValuableNode(func: Function, idd: Int) {
  val id = idd
  val function = func
  def getOut(inputWeights: Map[Edge, Double], inputValues: Map[Edge, Double]): Double = {
    val values: Seq[Value] = inputWeights.filter(_._1.to == id)
      .map(weight => Value(weight._2, inputValues.apply(weight._1))).toSeq
    function.getValue(values)
  }

  def deltaE(out: Double, target: Double, fromDeltaE: Seq[Double]): Double = {
    if (this.id == -1) { // OutputNode has id == -1
      return target - out
    }
    //delta of a node are all derivatives wrt to nodes value. Deduction of the weight is always 1
    function.derivative(out) * out * fromDeltaE.product
  }
}

case class ValueNode(func: Function, idd: Int) extends ValuableNode(func, idd) {


}
case class Edge(from: Int, to: Int)