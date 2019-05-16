package main.util

import akka.event.LoggingAdapter

trait Logging {
  val logger: Logger
  def debug(m: String): Unit = {
    logger.debug(m)
  }
  def info(m: String): Unit = {
    logger.info(m)
  }
  def warning(m: String): Unit = {
    logger.warning(m)
  }
  def error(m: String): Unit = {
    logger.error(m)
  }
}

case class Logger(logger: Option[LoggingAdapter]) {
  def debug(msg: String): Unit = {
    logger.map(_.debug(msg))
  }
  def info(msg: String): Unit = {
    logger.map(_.info(msg))
  }
  def warning(msg: String): Unit = {
    logger.map(_.warning(msg))
  }
  def error(msg: String): Unit = {
    logger.map(_.error(msg))
  }
}
