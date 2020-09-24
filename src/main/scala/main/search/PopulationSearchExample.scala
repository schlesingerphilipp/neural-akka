package main.search

import main.data.{DataPoint, ExampleData}
import main.neuralnet.domain.{GenerationService, _}

import scala.annotation.tailrec


case class PopulationSearchExample(train: Seq[DataPoint], validation: Seq[DataPoint],  mutationParameters: MutationParameters,
                            modelParameters: ModelParameters)  {


}
object PopulationSearchExample  extends App {
  /*val data = ExampleData.make(Math.random(), 5,100)
  val train = data.slice(0, (data.length * 0.7).toInt)
  val validate = data.slice((data.length * 0.7).toInt, data.length)
  val mutation = MutationParameters(0.1,0.1,0.1,0.1)
  val model = ModelParameters(3, 3, 1, 3)
  val modelOne = new GenerationService().generateModel(model)
  val modelMany = PopulationSearchExample(train, validate, mutation, model).localSearch(3, 100)
  val oneFit = modelOne.train(train).getMSE(validate)
  val manyFit = modelMany.train(train).getMSE(validate)
  System.out.println(s"oneFit ${oneFit} | manyFit ${manyFit}")*/


}