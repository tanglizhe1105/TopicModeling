/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.topicModeling

import java.text.BreakIterator

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, Logging, SparkConf, SparkContext}
import scopt.OptionParser

import scala.collection.mutable
import scala.reflect.runtime.universe._

/**
 * Abstract class for parameter case classes.
 * This overrides the [[toString]] method to print all case class fields by name and value.
  *
  * @tparam T  Concrete parameter class.
 */
abstract class AbstractParams[T: TypeTag] {

  private def tag: TypeTag[T] = typeTag[T]

  /**
   * Finds all case class fields in concrete class instance, and outputs them in JSON-style format:
   * {
   *   [field name]:\t[field value]\n
   *   [field name]:\t[field value]\n
   *   ...
   * }
   */
  override def toString: String = {
    val tpe = tag.tpe
    val allAccessors = tpe.declarations.collect {
      case m: MethodSymbol if m.isCaseAccessor => m
    }
    val mirror = runtimeMirror(getClass.getClassLoader)
    val instanceMirror = mirror.reflect(this)
    allAccessors.map { f =>
      val paramName = f.name.toString
      val fieldMirror = instanceMirror.reflectField(f)
      val paramValue = fieldMirror.get
      s"  $paramName:\t$paramValue"
    }.mkString("{\n", ",\n", "\n}")
  }
}

/**
 * An example Latent Dirichlet Allocation (LDA) app. Run with
 * {{{
 * ./bin/run-example mllib.LDAExample [options] <input>
 * }}}
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object LDAExample {

  private case class Params(
                             corpusSize: Int = 0,
                             windowSize: Int = 8000,
                             input: Seq[String] = Seq.empty,
                             k: Int = 20,
                             maxIterations: Int = 10,
                             maxInnerIterations: Int = 5,
                             docConcentration: Double = 0.01,
                             topicConcentration: Double = 0.01,
                             vocabSize: Int = 10000,
                             optimizer: String = "online",
                             partitions: Int = 2,
                             logLevel: String = "info"
                           ) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("LDAExample") {
      head("LDAExample: an example LDA app for plain text data.")
      opt[Int]("corpusSize")
        .text(s"window size. default: ${defaultParams.corpusSize}")
        .action((x, c) => c.copy(corpusSize = x))
      opt[Int]("windowSize")
        .text(s"window size. default: ${defaultParams.windowSize}")
        .action((x, c) => c.copy(windowSize = x))
      opt[Int]("k")
        .text(s"number of topics. default: ${defaultParams.k}")
        .action((x, c) => c.copy(k = x))
      opt[Int]("maxIterations")
        .text(s"number of iterations of learning. default: ${defaultParams.maxIterations}")
        .action((x, c) => c.copy(maxIterations = x))
      opt[Int]("maxInnerIterations")
        .text(s"number of inner iterations of learning. default: ${defaultParams.maxInnerIterations}")
        .action((x, c) => c.copy(maxInnerIterations = x))
      opt[Double]("docConcentration")
        .text(s"amount of topic smoothing to use (> 1.0) (-1=auto)." +
          s"  default: ${defaultParams.docConcentration}")
        .action((x, c) => c.copy(docConcentration = x))
      opt[Double]("topicConcentration")
        .text(s"amount of term (word) smoothing to use (> 1.0) (-1=auto)." +
          s"  default: ${defaultParams.topicConcentration}")
        .action((x, c) => c.copy(topicConcentration = x))
      opt[Int]("vocabSize")
        .text(s"number of distinct word types to use, chosen by frequency. (-1=all)" +
          s"  default: ${defaultParams.vocabSize}")
        .action((x, c) => c.copy(vocabSize = x))
      opt[String]("optimizer")
        .text(s"available optimizer are online and gibbs, default: ${defaultParams.optimizer}")
        .action((x, c) => c.copy(optimizer = x))
      opt[Int]("partitions")
        .text(s"Minimum edge partitions, default: ${defaultParams.partitions}")
        .action((x, c) => c.copy(partitions = x))
      opt[String]("logLevel")
        .text(s"Log level, default: ${defaultParams.logLevel}")
        .action((x, c) => c.copy(logLevel = x))
      arg[String]("<input>...")
        .text("input paths (directories) to plain text corpora." +
          "  Each text file line should hold 1 document.")
        .unbounded()
        .required()
        .action((x, c) => c.copy(input = c.input :+ x))
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      parser.showUsageAsError
      sys.exit(1)
    }
  }

  //    private def createOptimizer(params: Params, lineRdd: RDD[Int], columnRdd: RDD[Int]):LDAOptimizer = {
  private def createOptimizer(params: Params): LDAOptimizer = {
    params.optimizer match {
      case "online" => val optimizer = new OnlineLDAOptimizer
        optimizer
      case _ =>
        throw new IllegalArgumentException(s"available optimizers are em, online and gibbs, but got ${params.optimizer}")
    }
  }

  /**
    * run LDA
    *
    * @param params
    */
  private def run(params: Params) {
    val conf = new SparkConf()
      .setAppName(s"LDAExample with $params")
      .set("spark.locality.wait", "0s")
      .set("spark.memory.fraction", "0.5")
      .set("spark.memory.storageFraction", "0.5")
    val sc = new SparkContext(conf)

    val logLevel = Level.toLevel(params.logLevel, Level.INFO)
    Logger.getRootLogger.setLevel(logLevel)
    println(s"Setting log level to $logLevel")

    // Load documents, and prepare them for LDA.
    val preprocessStart = System.nanoTime()
    var docs = sc.objectFile[(Long, Vector)](params.input(0))
    docs = docs.coalesce(params.partitions, true)
    docs.cache()
    docs.count()
    val actualCorpusSize = params.corpusSize
    val actualVocabSize = docs.first()._2.toDense.size
    val actualNumTokens = docs.map(_._2.asInstanceOf[SparseVector].values.sum).take(actualCorpusSize).reduce(_ + _)
    val preprocessElapsed = (System.nanoTime() - preprocessStart) / 1e9

    println()
    println(s"[Corpus summary:]")
    println(s"[\t Training set size: $actualCorpusSize documents]")
    println(s"[\t Vocabulary size: $actualVocabSize terms]")
    println(s"[\t Training set size: $actualNumTokens tokens]")
    println(s"[\t Preprocessing time: $preprocessElapsed sec]")
    println()

    // Run LDA.
    val lda = new LDA()
    lda.setK(params.k)
      .setCorpusSize(actualCorpusSize)
      .setWindowSize(params.windowSize)
      .setPartition(params.partitions)
      .setVocabSeze(actualVocabSize)
      .setMaxIterations(params.maxIterations)
      .setDocConcentration(params.docConcentration)
      .setTopicConcentration(params.topicConcentration)
      .setOptimizer(createOptimizer(params))


    val startTime = System.nanoTime()
    val ldaModel = lda.run(docs)
    val elapsed = (System.nanoTime() - startTime) / 1e9

    println(s"Finished training LDA model using ${lda.getOptimizer.getClass.getName}")

    sc.stop()
  }
}


object HDPExample {

  private case class Params(
                             corpusSize: Int = 0,
                             windowSize: Int = 8000,
                             K: Int = 15,
                             T: Int = 150,
                             input: Seq[String] = Seq.empty,
                             maxIterations: Int = 10,
                             maxInnerIterations: Int = 5,
                             docConcentration: Double = 0.01,
                             topicConcentration: Double = 0.01,
                             vocabSize: Int = 10000,
                             optimizer: String = "online",
                             partitions: Int = 2,
                             logLevel: String = "info"
                           ) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("LDAExample") {
      head("LDAExample: an example LDA app for plain text data.")
      opt[Int]("corpusSize")
        .text(s"window size. default: ${defaultParams.corpusSize}")
        .action((x, c) => c.copy(corpusSize = x))
      opt[Int]("windowSize")
        .text(s"window size. default: ${defaultParams.windowSize}")
        .action((x, c) => c.copy(windowSize = x))
      opt[Int]("K")
        .text(s"number of topics. default: ${defaultParams.K}")
        .action((x, c) => c.copy(K = x))
      opt[Int]("T")
        .text(s"number of topics. default: ${defaultParams.T}")
        .action((x, c) => c.copy(T = x))
      opt[Int]("maxIterations")
        .text(s"number of iterations of learning. default: ${defaultParams.maxIterations}")
        .action((x, c) => c.copy(maxIterations = x))
      opt[Int]("maxInnerIterations")
        .text(s"number of inner iterations of learning. default: ${defaultParams.maxInnerIterations}")
        .action((x, c) => c.copy(maxInnerIterations = x))
      opt[Double]("docConcentration")
        .text(s"amount of topic smoothing to use (> 1.0) (-1=auto)." +
          s"  default: ${defaultParams.docConcentration}")
        .action((x, c) => c.copy(docConcentration = x))
      opt[Double]("topicConcentration")
        .text(s"amount of term (word) smoothing to use (> 1.0) (-1=auto)." +
          s"  default: ${defaultParams.topicConcentration}")
        .action((x, c) => c.copy(topicConcentration = x))
      opt[String]("optimizer")
        .text(s"available optimizer are online and gibbs, default: ${defaultParams.optimizer}")
        .action((x, c) => c.copy(optimizer = x))
      opt[Int]("partitions")
        .text(s"Minimum edge partitions, default: ${defaultParams.partitions}")
        .action((x, c) => c.copy(partitions = x))
      opt[String]("logLevel")
        .text(s"Log level, default: ${defaultParams.logLevel}")
        .action((x, c) => c.copy(logLevel = x))
      arg[String]("<input>...")
        .text("input paths (directories) to plain text corpora." +
          "  Each text file line should hold 1 document.")
        .unbounded()
        .required()
        .action((x, c) => c.copy(input = c.input :+ x))
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      parser.showUsageAsError
      sys.exit(1)
    }
  }

  /**
    * run LDA
    *
    * @param params
    */
  private def run(params: Params) {
    val conf = new SparkConf()
      .setAppName(s"HDPExample with $params")
      .set("spark.locality.wait", "0s")
      .set("spark.memory.fraction", "0.5")
      .set("spark.memory.storageFraction", "0.5")
    val sc = new SparkContext(conf)

    val logLevel = Level.toLevel(params.logLevel, Level.INFO)
    Logger.getRootLogger.setLevel(logLevel)
    println(s"Setting log level to $logLevel")

    // Load documents, and prepare them for LDA.
    val preprocessStart = System.nanoTime()
    var docs = sc.objectFile[(Long, Vector)](params.input(0))
    docs = docs.coalesce(params.partitions, true)
    docs.cache()
    docs.count()
    val actualCorpusSize = params.corpusSize
    val actualVocabSize = docs.first()._2.toDense.size
    val actualNumTokens = docs.map(_._2.asInstanceOf[SparseVector].values.sum).reduce(_ + _)
    val preprocessElapsed = (System.nanoTime() - preprocessStart) / 1e9

    println()
    println(s"[Corpus summary:]")
    println(s"[\t Training set size: $actualCorpusSize documents]")
    println(s"[\t Vocabulary size: $actualVocabSize terms]")
    println(s"[\t Training set size: $actualNumTokens tokens]")
    println(s"[\t Preprocessing time: $preprocessElapsed sec]")
    println()

    // Run HDP.
    val windowSize = params.windowSize
    val corpusSize = params.corpusSize
    val vocabSize = actualVocabSize
    val partition = params.partitions
    val maxIterations = params.maxIterations
    val K = params.K
    val T = params.T

    val state = new OnlineHDPOptimizer(corpusSize, windowSize, vocabSize, K, T)
    var windowCount = 0
    while ((windowCount + 1) * windowSize < corpusSize) {
      val begin = windowCount * windowSize
      val end = (windowCount + 1) * windowSize
      var batch = docs.filter(x => x._1 >= begin && x._1 < end)
      batch = batch.partitionBy(new HashPartitioner(partition))
      batch.cache()
      batch.count()

      var iter = 0
      val iterationTimes = Array.fill[Double](maxIterations)(0)
      val iterationPer = Array.fill[(Double, Double)](maxIterations)((0.0D, 0.0D))
      while (iter < maxIterations) {
        println(s"[windowcount+iter+maxiter: $windowCount $iter $maxIterations]")
        val start = System.nanoTime()
        val (docScore, tokenCount) = state.update_chunk(batch)
        val elapsedSeconds = (System.nanoTime() - start) / 1e9
        iterationTimes(iter) = elapsedSeconds

        var topicScore = 0D
        if(iter % 5 == 0){
          topicScore= state.topicPerplexity()
        }
        iterationPer(iter) = (docScore, topicScore)
        iter += 1
      }
      println(s"[windowcount+itertime: $windowCount ${iterationTimes.mkString(" ")}]")
      println(s"[windowcount+iterper: $windowCount ${iterationPer.mkString(" ")}]")

      batch.unpersist()
      windowCount += 1
    }


    sc.stop()
  }

}