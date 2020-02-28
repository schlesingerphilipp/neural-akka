package main.search

import main.data.{DataPoint, ExampleData}
import main.neuralnet.domain.{GenerationService, _}

import scala.annotation.tailrec


case class PopulationSearch(train: Seq[DataPoint], validation: Seq[DataPoint],  mutationParameters: MutationParameters,
                            modelParameters: ModelParameters)  {

  def localSearch(explorationSteps: Int, explorationVolume: Int): Model = {

    val initial = GenerationService.generateModel(modelParameters)
    val step = for (i<- 0 until explorationVolume) yield MutationService.mutate(initial, mutationParameters)
    searchStep(step, explorationSteps)
  }

  @tailrec
  private def searchStep(last: Seq[Model], explorationSteps: Int): Model = {
    System.out.println(f"searchSteps left ${explorationSteps}")
    if (explorationSteps == 0) {
      return last.head
    }
    val step = last.map(MutationService.mutate(_, mutationParameters))
    val next = (step.map(_.train(train)) ++ last).sortBy(_.getMSE(validation)).slice(0, (step.length + last.length) / 2)
    searchStep(next, explorationSteps - 1)
  }
}
object PopulationSearchExample  extends App {
  val data = ExampleData.make(Math.random(), 5,100)
  val train = data.slice(0, (data.length * 0.7).toInt)
  val validate = data.slice((data.length * 0.7).toInt, data.length)
  val mutation = MutationParameters(0.1,0.1,0.1,0.1)
  val model = ModelParameters(3, 3, 1, 3)
  val modelOne = GenerationService.generateModel(model)
  val modelMany = PopulationSearch(train, validate, mutation, model).localSearch(3, 100)
  val oneFit = modelOne.train(train).getMSE(validate)
  val manyFit = modelMany.train(train).getMSE(validate)
  System.out.println(s"oneFit ${oneFit} | manyFit ${manyFit}")


}