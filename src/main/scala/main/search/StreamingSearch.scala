package main.search

import main.data.DataPipeline
import main.neuralnet.domain.NextLayeredNetwork

import scala.annotation.tailrec
class StreamingSearch
object StreamingSearch {

  @tailrec
  def trainUntil(model: NextLayeredNetwork, dp: DataPipeline, trainDataPart: Double, counter: Int): NextLayeredNetwork = {
    if (counter >= dp.numberOfRows * trainDataPart) {
      return model
    }
    System.out.print("search step" + counter)
    val data = dp.load()
    trainUntil(model.train(data), dp, trainDataPart, counter + data.size)
  }
}
