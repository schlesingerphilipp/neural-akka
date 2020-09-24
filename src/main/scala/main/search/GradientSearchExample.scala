package main.search

import akka.actor.ActorSystem
import com.typesafe.config.ConfigFactory
import main.util.Logger

object GradientSearchExample {

  val config = ConfigFactory.load()
  val system: ActorSystem = ActorSystem(config.getString("akka-system"), config)
  implicit val logger = Logger(Option(system.log))
  /*def run(data: Data): Unit = {

    logger.info(s"meanSquareError:  ")
  }
*/
}



