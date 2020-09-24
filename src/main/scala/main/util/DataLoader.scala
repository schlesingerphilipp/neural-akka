package main.util

import main.data.DataPoint
import scala.sys.process.Process

object DataLoader {
  def selectRandomInput(mustContain: Seq[String]): Seq[String] = ???


  private def combineData(inputs: Seq[String]): String = {
    val sorted = inputs.sorted
    val preparationProcess = Process(f"python3 pg_loader.py ${sorted.reduce(_ + " " + _)}")
    preparationProcess.! match {
      case 0 => sorted.reduce(_ + "_" + _) + ".csv"
      case _ => throw new Exception("Data Preparation Failed")
    }
  }

  private def toDataPoint(str: Seq[String], targetIndex: Integer): DataPoint = {
    val ds = str.map(_.toDouble)
    DataPoint(ds.apply(targetIndex), ds.drop(targetIndex))
  }


  private def getTargetIndex(selected: Seq[String], target: String, field: Integer): Integer = {
    (selected.indexOf(target) + 1) * 5 - field
  }


  private def loadData(filePath: String, targetIndex: Int): Seq[DataPoint] = {
    val bufferedSource = io.Source.fromFile(filePath)
    val data = for (line <- bufferedSource.getLines) yield line.split(",").map(_.trim).toSeq
    bufferedSource.close
    data.map(toDataPoint(_, targetIndex)).toSeq
  }

  /**
    * One currency is made of the 5 fields open, close, low, high, volume.
    * Depending on which of these fields of the currency the target is, the targetIndex varies
    * @param selected the list of selected currencies, which must contain the target
    * @param target the name of the currency
    * @param field open, close, low, high, volume. TODO scala enum import issues.
    * @return the index of the target in the list of currency fields
    */

  def prepareData(selected: Seq[String], target: String, field: Integer): Seq[DataPoint] = {
    val filePath = combineData(selected)
    val targetIndex = getTargetIndex(selected, target, field)
    loadData(filePath, targetIndex)
  }

}