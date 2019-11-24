package main.neuralnet.domain

trait MutationService {
  def mutate(model: Model, mutationParameters: MutationParameters, modelParameters: ModelParameters): Model
}

object LayeredNetworkMutationService extends MutationService {
  val random = new scala.util.Random
  override def mutate(model: Model, mutationParameters: MutationParameters, modelParameters: ModelParameters): Model = {
    model match {
      case l: LayeredNetwork =>
        l.mutate(m => if (mutationParameters.addNodeChance > Math.random()) addNode(m) else m)
          .mutate(m => if (mutationParameters.addNodeChance > Math.random()) removeNode(m) else m)
          .mutate(m => if (mutationParameters.addNodeChance > Math.random()) addLayer(m, modelParameters) else m)
          .mutate(m => if (mutationParameters.addNodeChance > Math.random()) removeLayer(m) else m)
    }
  }

  def addNode(model: LayeredNetwork): LayeredNetwork = {
    val maxId: Int = getMaxNodeId(model)
    val selectedLayer: Int = random.nextInt(model.layers.length -2) + 1
    val updatingLayer: Layer = model.layers.apply(selectedLayer)
    val previous: Layer = model.layers.apply(selectedLayer -1 )
    val newNodes: Seq[ValueNode] = Seq(ValueNode(Sigma(), maxId +1))
    val newWeights: Map[Edge, Double] = LayeredNetworkGenerationService.generateInputWeights(newNodes, previous)
    val updatedLayer: Layer = Layer(updatingLayer.nodes ++ newNodes, updatingLayer.inputWeights ++ newWeights)
    val followingLayer: Layer = model.layers.apply(selectedLayer + 1)
    val followingLayerNewWeights: Map[Edge, Double] = LayeredNetworkGenerationService
      .generateInputWeights(followingLayer.nodes, updatedLayer)
    val updatedFollowing: Layer = Layer(followingLayer.nodes, followingLayerNewWeights)
    LayeredNetwork(
      (model.layers.slice(0, selectedLayer)
        :+ updatedLayer
        :+ updatedFollowing)
        ++ model.layers.slice(selectedLayer +2, model.layers.length))
  }
  def removeNode(model: LayeredNetwork): LayeredNetwork = {
    val selectedLayer: Int = random.nextInt(model.layers.length -2) + 1
    val updatingLayer: Layer = model.layers.apply(selectedLayer)
    val selectedNode = updatingLayer.nodes.apply(random.nextInt(updatingLayer.nodes.length)).id
    val remainingWeights = updatingLayer.inputWeights.filter(_._1.from != selectedNode)
    val remainingNodes = updatingLayer.nodes.filter(_.id != selectedNode)
    val updatedLayer = Layer(remainingNodes,remainingWeights)
    val following = model.layers.apply(selectedLayer -1 )
    val followingRemainingWeights = following.inputWeights.filter(_._1.to != selectedNode)
    val followingUpdatedLayer =  Layer(following.nodes, followingRemainingWeights)
    LayeredNetwork(
      (model.layers.slice(0, selectedLayer)
        :+ updatedLayer
        :+ followingUpdatedLayer)
        ++ model.layers.slice(selectedLayer +2, model.layers.length))
  }
  def addLayer(model: LayeredNetwork, modelParameters: ModelParameters): LayeredNetwork = {
    val index: Int = random.nextInt(model.layers.length -2) + 1
    val nodeCount = modelParameters.minLayerVolume +
      random.nextInt(modelParameters.maxLayerVolume - modelParameters.minLayerVolume)
    val idCounter = getMaxNodeId(model) + 1
    val previous = model.layers.apply(index -1)
    val newLayer = LayeredNetworkGenerationService.generateHiddenLayer(previous, nodeCount, idCounter)
    val following = model.layers.apply(index)
    val newWeights: Map[Edge, Double] = LayeredNetworkGenerationService.generateInputWeights(newLayer.nodes, previous)
    val updatedFollowing = Layer(following.nodes, newWeights)
    LayeredNetwork(
      (model.layers.slice(0, index)
        :+ newLayer
        :+ updatedFollowing)
        ++ model.layers.slice(index +1, model.layers.length))
  }
  def removeLayer(model: LayeredNetwork): LayeredNetwork= {
    val index: Int = random.nextInt(model.layers.length -2) + 1
    val previous = model.layers.apply(index -1)
    val following = model.layers.apply(index +1)
    val newWeights: Map[Edge, Double] = LayeredNetworkGenerationService.generateInputWeights(following.nodes, previous)
    val updatedFollowing = Layer(following.nodes, newWeights)
    LayeredNetwork(
      (model.layers.slice(0, index)
        :+ updatedFollowing)
        ++ model.layers.slice(index +1, model.layers.length))
  }

  private def getMaxNodeId(model: LayeredNetwork): Int = {
    model.layers.flatMap(_.nodes).maxBy(_.id).id
  }
}
case class MutationParameters(addLayerChance: Double, removeLayerChance: Double, addNodeChance: Double, removeNodeChance: Double)
