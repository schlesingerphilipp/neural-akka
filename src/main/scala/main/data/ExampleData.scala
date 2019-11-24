package main.data

object ExampleData {
  def make(seed: Double, factors: Integer, sampleSize: Integer): Seq[DataPoint] = {
    def getXs(): Seq[Double] = {
      for (i <- 0 until factors) yield seed * Math.random()
    }
    def getPoint(ws:Seq[Double]): DataPoint = {
      val xs: Seq[Double] = getXs()
      val y: Double =  xs.zipAll(ws,0.0,0.0).map((a:(Double,Double)) =>a._1*a._2).foldLeft(0.0)(_ + _)
      DataPoint(y, xs)
    }
    val weights = for (i <- 0 until factors)
      yield Math.random()
    for (j <- 0 until sampleSize)
      yield getPoint(weights)
  }
}
case class ExampleData(seed: Double, factors: Integer, sampleSize: Integer) extends Data {
  val data: Seq[DataPoint] = ExampleData.make(seed, factors, sampleSize)
}
case class DataPoint(target: Double, features: Seq[Double])
trait Data{
  val data: Seq[DataPoint]
  override def toString(): String = {
    val seq: Seq[String] = for {
      i <- data
    } yield i.toString()
    seq.foldLeft("")(_ + " , " +  _)
  }
  def getTrainingTestSplit(trainingPortion: Double): (Seq[DataPoint], Seq[DataPoint]) = {
    data.splitAt(Math.round(data.length * trainingPortion).toInt)
  }
}
