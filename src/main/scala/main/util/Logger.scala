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

class TestAdapter extends LoggingAdapter {
  override def isErrorEnabled: Boolean = true

  override def isWarningEnabled: Boolean = true

  override def isInfoEnabled: Boolean = true

  override def isDebugEnabled: Boolean = true

  override protected def notifyError(message: String): Unit = println("[ERROR] " + message)

  override protected def notifyError(cause: Throwable, message: String): Unit = println("[ERROR] " + message + "\n" + cause.getMessage)

  override protected def notifyWarning(message: String): Unit = println("[WARN] " + message)

  override protected def notifyInfo(message: String): Unit = println("[INFO] " + message)

  override protected def notifyDebug(message: String): Unit = println("[DEBUG] " + message)
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
