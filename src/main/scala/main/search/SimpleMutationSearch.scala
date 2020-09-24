package main.search

import main.data.DataPoint
/*

object SimpleMutationSearch  extends App {

  def selectStrongest(modelData: Seq[ModelData]):Model = {
    modelData.head.model
  }

  def trainModel(model: ModelData): ModelData = {
    ModelData(model.model.train(model.data.train), model.data)
  }

  def mutate(iteration: Seq[ModelData]): Seq[ModelData] = ???

  def train(modelData: Seq[ModelData], stoppingCriteria: StoppingCriteria): Model = {
    val iteration: Seq[ModelData] = modelData.map(trainModel)
    stoppingCriteria.add(iteration.flatMap(md => md.data.test.map(
      data => Math.pow(md.model.predict(data.features) - data.target, 2))))
    if (stoppingCriteria.saturated(0.05)) {
      return selectStrongest(iteration)
    }
    val mutations = mutate(iteration)
    train(mutations, stoppingCriteria)
  }
}
case class Data(train: Seq[DataPoint], test: Seq[DataPoint], validate: Seq[DataPoint])
case class ModelData(model: Model, data: Data)
*/