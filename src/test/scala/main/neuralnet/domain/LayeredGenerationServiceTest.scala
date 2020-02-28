package main.neuralnet.domain

import org.scalatest.FlatSpec

import scala.annotation.tailrec

class LayeredGenerationServiceTest extends FlatSpec {

  "generateInputLayer" should "yield an input layer" in {
    val inputs: Int = 3
    val inputLayer: Layer = GenerationService.generateInputLayer(inputs)
    assert(inputLayer.nodes.length == 3, "Unexpected node count.")
    assert(inputLayer.inputWeights.size == 3, "Unexpected weight count.")
    assert(inputLayer.inputWeights.count(_._2 == 1.0) == inputs, "Not all weights were 1.0.")
    assert(inputLayer.inputWeights.count(_._1.from == null) == inputs, "Not all from edges are null")
  }


  "generateHiddenLayer" should "yield an hidden layer" in {
    val previousLayer: Layer = GenerationService.generateInputLayer(3)
    val nodeCount: Int = 3
    val idCounter: Int = 4
    val layer = GenerationService.generateHiddenLayer(previousLayer, nodeCount, idCounter)
    assert(layer.nodes.length == 3, "Unexpected number of nodes")
    assert(layer.inputWeights.size == previousLayer.nodes.length * layer.nodes.length, "Unexpected edge count.")
    assert(layer.inputWeights.forall(_._1.from != null), "At least one from edges is null")
    assert(layer.inputWeights.forall(_._1.to != null), "At least one to edges is null")
    assert(layer.inputWeights.forall(w => layer.nodes.map(_.id).contains(w._1.to)), "Not all to targets " +
      "are in the list of node ids of this layer")
  }

  "generateOutputLayer" should "yield an hidden layer" in {
    val previousLayer: Layer = GenerationService.generateInputLayer(3)

    val outLayer: Layer = GenerationService.generateOutputLayer(previousLayer);
    assert(outLayer.nodes.length == 1, "Unexpected node count.")
    assert(outLayer.inputWeights.size == previousLayer.nodes.length * outLayer.nodes.length, "Unexpected edge count.")
    assert(outLayer.nodes.forall(_.function.isInstanceOf[Sum]), "Not all out nodes are sum functions")
    assert(outLayer.inputWeights.forall(_._1.to == outLayer.nodes.head.id), "Not all edges go to the one outnode")
  }

  "generateInputWeights" should "yield expected connections" in {
    val nodes = Seq(ValueNode(Sigma(), 5), ValueNode(Sigma(), 6))
    val previous = GenerationService.generateInputLayer(5);
    val actual: Map[Edge, Double] = GenerationService.generateInputWeights(nodes, previous)
    val expected = Map(Edge(0,5) -> 0.0, Edge(1,5) -> 0.0, Edge(2,5) -> 0.0, Edge(3,5) -> 0.0, Edge(4,5) -> 0.0,
      Edge(0,6) -> 0.0, Edge(1,6) -> 0.0, Edge(2,6) -> 0.0, Edge(3,6) -> 0.0, Edge(4,6) -> 0.0 )
    assert(expected.keys.forall(actual.contains), "Not all Edges in Map")
  }

  "generateModel" should "generate a model according to parameters" in {
    val (inputs, hiddenLayers, min, max) = (3,3,3,5)
    val params = ModelParameters(inputs, hiddenLayers, min, max)
    val model: LayeredNetwork = GenerationService.generateModel(params).asInstanceOf[LayeredNetwork]
    assert(model.layers.length == 1 + hiddenLayers + 1, "There are more or less layers than expected")
    assert(model.layers.head.nodes.forall(_.function.isInstanceOf[Input]), "Head of layers should be the input " +
      "layer, where all nodes functions should be input functions")
    assert(model.layers.head.nodes.length == inputs, "There are not as many input nodes as there are inputs")
    TestUtil.assertAllLayersConnections(model.layers.tail, model.layers.head)

  }

}

protected object TestUtil {
  def assertConnections(from: Layer, to: Layer) = {

    assert(from.nodes.map(_.id).forall(fromId => to.inputWeights.keys.count(_.from == fromId) == to.nodes.length),
      "There are as many edges with the 'from' id as there are nodes in the 'to' layer")
    //
    assert(to.nodes.map(_.id).forall(toId => to.inputWeights.keys.count(_.to == toId) == from.nodes.length),
      "There are as many edges with the 'to' id as there are nodes in the 'from' layer")
    assert(to.nodes.map(_.id).forall(id => to.inputWeights.map(_._1.to).exists(_ == id)),
      "Every node 'to' id is present in the edges as the 'to' field of the edge")

    assert(from.nodes.map(_.id).forall(id => to.inputWeights.map(_._1.from).exists(_ == id)),
      "Every node 'from' id is present in the edges as the 'from' field of the edge")
    assert(to.inputWeights.size == from.nodes.length * to.nodes.length, "Fully connected there are n * m Edges")
  }

  @tailrec
  def assertAllLayersConnections(tail: Seq[Layer], head: Layer): Unit = {
    if (tail.isEmpty) {
      return Unit
    }
    assertConnections(head, tail.head)
    assertAllLayersConnections(tail.tail, tail.head)
  }
}