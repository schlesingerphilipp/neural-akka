package main.neuralnet.domain


object MutationService {
  val random = new scala.util.Random

  def mutateDefinitionFrom(model: LayeredNetwork, mutationParameters: MutationParameters): Seq[Int] = {
    val inputs = Seq(model.layers.head.nodes.length)
    val hidden = model.layers.slice(1,model.layers.length -1)
      .flatMap(l => {
        val mutations = generateMutation(mutationParameters)
        for (i <- 0 to mutations.layers) yield l.nodes.length + mutations.nodes
    })
    val outputs = Seq(1)
    inputs ++ hidden ++ outputs
  }

  def generateMutation(mutationParameters: MutationParameters): Mutations = {
    val addLayer = if (random.nextGaussian() < mutationParameters.addLayerChance) 1 else 0
    val removeLayer = if (random.nextGaussian() < mutationParameters.removeLayerChance) -1 else 0
    val addNode = if (random.nextGaussian() < mutationParameters.addNodeChance) 1 else 0
    val removeNode = if (random.nextGaussian() < mutationParameters.removeNodeChance) -1 else 0
    Mutations(1 + addLayer + removeLayer, addNode + removeNode)
  }

  def mutate(model: Model, mutationParameters: MutationParameters): Model = {
    model match {
      case l: LayeredNetwork => {
        val layerDefinition: Seq[Int] = mutateDefinitionFrom(l, mutationParameters)
        GenerationService.generateFromDefinition(layerDefinition)
      }
    }
  }

}
case class MutationParameters(addLayerChance: Double, removeLayerChance: Double,
                              addNodeChance: Double, removeNodeChance: Double)
case class Mutations(layers: Int, nodes: Int)
