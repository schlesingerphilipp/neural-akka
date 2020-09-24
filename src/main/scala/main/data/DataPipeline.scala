package main.data
import java.sql.{Connection, ResultSet}

import main.neuralnet.domain.NextDataPoint

import scala.annotation.tailrec

case class DataPipeline(target: Target, features: Seq[Feature], fetchSize: Int)(implicit conn: Connection){

  var targetRs: NextResultSet = loadTargets(target, fetchSize)
  targetRs.next()
  var featureRs: Map[String, NextResultSet] = loadFeatures(features, targetRs.getTime(), 50)
  val firstFeatureRow: Map[String,  Seq[Double]] = loadRowWithoutTimeConstraint()
  val firstDP = NextDataPoint(targetRs.get().head, firstFeatureRow)
  val numberOfRows: Int = setNumberOfRows()


  def loadRowWithoutTimeConstraint(): Map[String, Seq[Double]] = {
    featureRs.map({case (k: String,v: NextResultSet) => {
      if (v.next()) {
        (k, v.get())
      } else {
        (k,v.getZeroRow())
      }
    }})
  }

  def load(): Seq[NextDataPoint] = {
    //We already loaded the first Target
    // We load the first feature row, which is before first target by sql select, or zeros if no row is before
    loadNext(Seq(firstDP), fetchSize)
  }


  @tailrec
  private def loadNext(acc: Seq[NextDataPoint], rowsToLoad: Int): Seq[NextDataPoint] = {
    if (rowsToLoad == 0 || !targetRs.next()) {
      return acc
    }
    val time = targetRs.getTime()
    val targetRow = targetRs.get()
    val features: Map[String, Seq[Double]] = featureRs.map({case (k: String,v: NextResultSet) => {
      if (v.next() && time <= v.getTime()) {
        (k, v.get())
      } else {
        (k,acc.head.features.apply(k))
      }
    }})
    loadNext(acc :+ NextDataPoint(targetRow.head, features), rowsToLoad - 1)
  }


  private def selectFromAfter(table: String, fields: Seq[String], dateField: String,
                              after: Double, fetchSize: Int): NextResultSet = {
    val st = conn.createStatement
    st.setFetchSize(fetchSize)
    val selected = fields.reduce((a,b) => "%s, %s".format(a,b)) + ", %s".format(dateField)
    SqlResultSet(st.executeQuery("SELECT %s FROM %s WHERE %s >= %s order by %s ASC;"
      .format(selected, table, dateField, after, dateField)))
  }


  private def loadFeatures(features: Seq[Feature], notBefore: Double, fetchSize: Int): Map[String, NextResultSet] = {
    features.map {
      case feature: SqlFeature =>
        val results = selectFromAfter(feature.table, feature.fields, feature.dateField, notBefore + feature.timeshift, fetchSize)
        (feature.table, results)
    }.toMap
  }

  private def loadTargets(target: Target, fetchSize: Int): NextResultSet = {
    target match {
      case t: SqlTarget =>
       selectFromAfter(t.table, Seq(t.field), t.dateField, 0.0, fetchSize)
    }
  }

  private def setNumberOfRows(): Int = {
    target match {
      case t: SqlTarget => {
        val st = conn.createStatement
        val resp = st.executeQuery("SELECT Count(*) FROM %s;".format(t.table))
        resp.next()
        return resp.getInt(1)
      }
    }
    0
  }

}
class Target(source: String)
case class SqlTarget(table: String, field: String, dateField: String) extends Target(table)
class Feature(source: String)
case class SqlFeature(table: String, fields: Seq[String], dateField: String, timeshift: Double) extends Feature(table)
trait NextResultSet {
  def next(): Boolean
  def get(): Seq[Double]
  def rowSize(): Int
  def getTime(): Double
  def getZeroRow(): Seq[Double]
}
case class SqlResultSet(resultSet: ResultSet) extends NextResultSet{
  override def next(): Boolean = resultSet.next()
  override def get(): Seq[Double] = {
    //All columns except last column, which is the time. ResultSet starts at index 1, instead of 0
    for(i <- 1 until rowSize()) yield resultSet.getDouble(i)
  }
  override def rowSize(): Int = resultSet.getMetaData.getColumnCount

  override def getTime(): Double = {
    //Last column is the time
    resultSet.getDouble(rowSize())
  }

  override def getZeroRow(): Seq[Double] = {
    Seq.fill(rowSize() -1){0.0}
  }

}