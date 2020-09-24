name := "akka-nn-evolution"

version := "1.0"

scalaVersion := "2.12.6"

lazy val akkaVersion = "2.5.18"
lazy val scalatestVersion = "3.0.5"
lazy val postgresVersion = "42.2.16"
lazy val root = project in file(".")

libraryDependencies ++= Seq(
  "com.typesafe.akka" %% "akka-actor" % akkaVersion,
  "com.typesafe.akka" %% "akka-testkit" % akkaVersion,
  "com.typesafe.akka" %% "akka-slf4j" % akkaVersion,
  "ch.qos.logback" % "logback-classic" % "1.2.3",
  "com.typesafe" % "config" % "1.3.2" ,
  "org.scalactic" %% "scalactic" % scalatestVersion,
  "org.scalatest" %% "scalatest" % scalatestVersion % Test,
  "org.postgresql" % "postgresql" % postgresVersion
)
