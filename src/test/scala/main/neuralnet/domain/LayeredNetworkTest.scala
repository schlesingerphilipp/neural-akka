package main.neuralnet.domain

import main.data.ExampleData
import org.scalatest._

/*class LayeredNetworkTest extends FlatSpec with Matchers {
  "A Training " should "decrease the MSE" in {
    val net = GenerationService.generateModel(ModelParameters(3, 2, 3, 6))
    val data = ExampleData(Math.random(), 3, 100)
    val (training, test) = data.getTrainingTestSplit(0.7)
    val trainedNet = net.train(training)
    val untrainedMSE = net.predict(test.map(_.target))
    val trainedMSE = trainedNet.predict(test.map(_.target))
    assert(untrainedMSE > trainedMSE, "The MSE did not decrease in the training.")
  }
}*/