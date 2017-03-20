package com.jaarrechea.deeplearning4j;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Clasificación de materiales
 *
 * Ejemplo de clasificación de fotos de dos tipos de materiales: madera o metal.
 *
 * Las imágenes han sido seleccionadas al azar de Google.
 *
 */

public class ClasificacionMateriales {
  protected static final Logger log = LoggerFactory
      .getLogger(ClasificacionMateriales.class);
  protected static int height = 100;
  protected static int width = 100;

  // Imágenes en color
  protected static int channels = 3;

  // Ejemplo materiales
  protected static int numExamples = 120; // 60 de madera y 60 de metal
  protected static int numLabels = 2;
  protected static int batchSize = 60;

  protected static long seed = 42;
  protected static Random rng = new Random(seed);
  protected static int listenerFreq = 1;
  protected static int iterations = 1;
  protected static int epochs = 50;
  protected static double splitTrainTest = 0.8; // 80% para entrenar y 20% para
                                                // test
  protected static int nCores = 2;

  // Por defecto, el modelo de multicapas elegido es AlexNet
  protected static String modelType = "AlexNet";

  public void run(String[] args) throws Exception {

    log.info("Carga de datos....");
    
    // Etiquetador que considera etiqueta de cada imagen el nombre del 
    // directorio donde se encuentra el archivo de la imagen.
    ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    
    // Directorio donde se encuentran las imágenes
    File mainPath = new File(System.getProperty("user.dir"),
        "src/main/resources/materiales/");
    
    // Define las dataset básicos que se van a utilizar limitando el tipo de
    // imágenes
    FileSplit fileSplit = new FileSplit(mainPath,
        NativeImageLoader.ALLOWED_FORMATS, rng);

    // Utilidad que elige al azar las imágenes que debe analizar, igualando el
    // número de las mismas por cada clase. Limita y equilibra el contenido de 
    // cada lote.
    BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker,
        numExamples, numLabels, batchSize);

    // Array en el que almacenan los datos que van a servir para entrenamiento
    // y los que van a servir para testeo (pruebas y validaciones)
    InputSplit[] inputSplit = fileSplit.sample(pathFilter,
        numExamples * (splitTrainTest), numExamples * (1 - splitTrainTest));
    InputSplit trainData = inputSplit[0];
    InputSplit testData = inputSplit[1];

    // Diversas transformaciones sobre las imágenes, creando un gran dataset
    // sobre el que entrenar
    ImageTransform flipTransform1 = new FlipImageTransform(rng);
    ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
    ImageTransform warpTransform = new WarpImageTransform(rng, 42);
    List<ImageTransform> transforms = Arrays.asList(
        new ImageTransform[] { flipTransform1, warpTransform, flipTransform2 });

    // Normaliza imágenes y genera un gran dataset sobre el que entrenar
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

    log.info("Construyendo modelo...");

    // Switch que inicializa el modelo multicapa según el valor solicitado
    MultiLayerNetwork network;
    switch (modelType) {
    case "LeNet":
      network = lenetModel();
      break;
    case "AlexNet":
      network = alexnetModel();
      break;
    case "custom":
      network = customModel();
      break;
    default:
      throw new InvalidInputTypeException(
          "El modelo proporcionado es incorrecto.");
    }
    network.init();
    network.setListeners(new ScoreIterationListener(listenerFreq));

    // Lector de imágenes que las inicializa, carga y convierte
    ImageRecordReader recordReader = new ImageRecordReader(height, width,
        channels, labelMaker);
    // Generador que sólo carga un lote cada vez en memoria para ahorrar memoria
    DataSetIterator dataIter;
    // Asegura que los datos pasan todos los epochs
    MultipleEpochsIterator trainIter;

    log.info("Entrenando modelo...");
    // Entrenado sin transformaciones
    recordReader.initialize(trainData, null);
    dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1,
        numLabels);
    scaler.fit(dataIter);
    dataIter.setPreProcessor(scaler);
    trainIter = new MultipleEpochsIterator(epochs, dataIter, nCores);
    network.fit(trainIter);

    // Entrenando transformaciones
    for (ImageTransform transform : transforms) {
      System.out.print("\nEntrenando transformación: "
          + transform.getClass().toString() + "\n\n");
      recordReader.initialize(trainData, transform);
      dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1,
          numLabels);
      scaler.fit(dataIter);
      dataIter.setPreProcessor(scaler);
      trainIter = new MultipleEpochsIterator(epochs, dataIter, nCores);
      network.fit(trainIter);
    }

    log.info("Evaluando modelo...");
    recordReader.initialize(testData);
    dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1,
        numLabels);
    scaler.fit(dataIter);
    dataIter.setPreProcessor(scaler);
    Evaluation eval = network.evaluate(dataIter);
    log.info(eval.stats(true));

    // Ejemplo de cómo predice resultados el modelo entrenado
    dataIter.reset();
    DataSet testDataSet = dataIter.next();
    String expectedResult = testDataSet.getLabelName(0);
    List<String> predict = network.predict(testDataSet);
    String modelResult = predict.get(0);
    System.out.print("\nPara un ejemplo concreto etiquetado como "
        + expectedResult + " el modelo predijo " + modelResult + "\n\n");

    log.info("****************Ejemplo finalizado********************");
  }

  private ConvolutionLayer convInit(String name, int in, int out, int[] kernel,
      int[] stride, int[] pad, double bias) {
    return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in)
        .nOut(out).biasInit(bias).build();
  }

  private ConvolutionLayer conv3x3(String name, int out, double bias) {
    return new ConvolutionLayer.Builder(new int[] { 3, 3 }, new int[] { 1, 1 },
        new int[] { 1, 1 }).name(name).nOut(out).biasInit(bias).build();
  }

  private ConvolutionLayer conv5x5(String name, int out, int[] stride,
      int[] pad, double bias) {
    return new ConvolutionLayer.Builder(new int[] { 5, 5 }, stride, pad)
        .name(name).nOut(out).biasInit(bias).build();
  }

  private SubsamplingLayer maxPool(String name, int[] kernel) {
    return new SubsamplingLayer.Builder(kernel, new int[] { 2, 2 }).name(name)
        .build();
  }

  private DenseLayer fullyConnected(String name, int out, double bias,
      double dropOut, Distribution dist) {
    return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias)
        .dropOut(dropOut).dist(dist).build();
  }

  public MultiLayerNetwork lenetModel() {
    /**
     * Modelo revisado Lenet. Desarrollo planteao por  ramgo2. 
     * 
     * Consultar:
     * https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
     * 
     **/
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed).iterations(iterations).regularization(false).l2(0.005) // tried
                                                                           // 0.0001,
                                                                           // 0.0005
        .activation(Activation.RELU).learningRate(0.0001) // tried 0.00001,
                                                          // 0.00005, 0.000001
        .weightInit(WeightInit.XAVIER)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(Updater.RMSPROP).momentum(0.9).list()
        .layer(0,
            convInit("cnn1", channels, 50, new int[] { 5, 5 },
                new int[] { 1, 1 }, new int[] { 0, 0 }, 0))
        .layer(1, maxPool("maxpool1", new int[] { 2, 2 }))
        .layer(2,
            conv5x5("cnn2", 100, new int[] { 5, 5 }, new int[] { 1, 1 }, 0))
        .layer(3, maxPool("maxpool2", new int[] { 2, 2 }))
        .layer(4, new DenseLayer.Builder().nOut(500).build())
        .layer(5,
            new OutputLayer.Builder(
                LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(numLabels).activation(Activation.SOFTMAX).build())
        .backprop(true).pretrain(false)
        .setInputType(InputType.convolutional(height, width, channels)).build();

    return new MultiLayerNetwork(conf);

  }

  public MultiLayerNetwork alexnetModel() {
    /**
     * Modelo AlexNet obtenido a partir del papep original
     * ImageNet Classification with Deep Convolutional Neural Networks
     * 
     * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
     **/

    double nonZeroBias = 1;
    double dropOut = 0.5;

    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed).weightInit(WeightInit.DISTRIBUTION)
        .dist(new NormalDistribution(0.0, 0.01)).activation(Activation.RELU)
        .updater(Updater.NESTEROVS).iterations(iterations)
        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize
                                                                            // to
                                                                            // prevent
                                                                            // vanishing
                                                                            // or
                                                                            // exploding
                                                                            // gradients
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .learningRate(1e-2).biasLearningRate(1e-2 * 2)
        .learningRateDecayPolicy(LearningRatePolicy.Step).lrPolicyDecayRate(0.1)
        .lrPolicySteps(100000).regularization(true).l2(5 * 1e-4).momentum(0.9)
        .miniBatch(false).list()
        .layer(0,
            convInit("cnn1", channels, 96, new int[] { 11, 11 },
                new int[] { 4, 4 }, new int[] { 3, 3 }, 0))
        .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
        .layer(2, maxPool("maxpool1", new int[] { 3, 3 }))
        .layer(3,
            conv5x5("cnn2", 256, new int[] { 1, 1 }, new int[] { 2, 2 },
                nonZeroBias))
        .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
        .layer(5, maxPool("maxpool2", new int[] { 3, 3 }))
        .layer(6, conv3x3("cnn3", 384, 0))
        .layer(7, conv3x3("cnn4", 384, nonZeroBias))
        .layer(8, conv3x3("cnn5", 256, nonZeroBias))
        .layer(9, maxPool("maxpool3", new int[] { 3, 3 }))
        .layer(10,
            fullyConnected("ffn1", 4096, nonZeroBias, dropOut,
                new GaussianDistribution(0, 0.005)))
        .layer(11,
            fullyConnected("ffn2", 4096, nonZeroBias, dropOut,
                new GaussianDistribution(0, 0.005)))
        .layer(12,
            new OutputLayer.Builder(
                LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output")
                    .nOut(numLabels).activation(Activation.SOFTMAX).build())
        .backprop(true).pretrain(false)
        .setInputType(InputType.convolutional(height, width, channels)).build();

    return new MultiLayerNetwork(conf);

  }

  public static MultiLayerNetwork customModel() {
    /**
     * Utilizar este método para construir un modelo propio.
     **/
    return null;
  }

  public static void main(String[] args) throws Exception {
    new ClasificacionMateriales().run(args);
  }

}
