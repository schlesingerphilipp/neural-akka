package main.search
import main.NeuralNet.Network
import main.data.Data
class PopulationSearch(data: Data) {
  val training, test = data.getTrainingTestSplit(0.7)
  def fitnessTest(individual: Network): Double = ???
  def createPopulation(): Population = ???
  def breed(pop1: Population, pop2: Population): (Population, Population) = ???
}

class Population(members: Seq[Network]) {}


