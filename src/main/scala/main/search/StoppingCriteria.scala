package main.search

case class StoppingCriteria(var errors: Seq[Double] = Seq.empty) {
  def add(es: Seq[Double]) = {
    errors = errors ++ es
  }
  def saturated(improvementThreshold: Double): Boolean = {
    if (errors.length < 40) {
      return false
    }
    def tenPercent = errors.length / 10
    def offset = if (tenPercent < 20) 20 else tenPercent
    def last = errors.slice(errors.length - offset, errors.length).sum
    def comparision = errors.slice(errors.length - offset * 2, errors.length - offset).sum
    1 - (last / comparision) < improvementThreshold
  }
}
