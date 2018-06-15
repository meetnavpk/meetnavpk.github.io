package io.github.meetnavpk.mdlc.mlp.dbn;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class DeepBeliefNetwork {

    private final static Logger LOGGER = LoggerFactory.getLogger(DeepBeliefNetwork.class);

    public static void main(final String... args) throws Exception {
//        final int numRows = 28,
//                numColumns = 28,
//                outputNum = 10,
//                batchSize = 100,
//                iterations = 10,
//                rngSeed = 123,// random number seed for reproducibility
//                listenerFreq = batchSize / 5;
//
//        LOGGER.info("Load data....");
//
//        //Get the DataSetIterators:
//        MnistDownloader.download(); //Workaround for download location change since 0.9.1 release
//        final DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
//        final DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
//
//        LOGGER.info("Build model....");
//
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(rngSeed)
//                .gradientNormalization(
//                        GradientNormalization.ClipElementWiseAbsoluteValue)
//                .gradientNormalizationThreshold(1.0)
////                .iterations(iterations)
//                .weightInit(WeightInit.XAVIER)
////                .updater(Updater.NESTEROVS)
//                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
//                .list()
//                .layer(0,
//                        new RBM.Builder().nIn(numRows * numColumns).nOut(500)
//                                .weightInit(WeightInit.XAVIER)
//                                .lossFunction(LossFunction.MCXENT)
//                                .visibleUnit(RBM.VisibleUnit.BINARY)
//                                .hiddenUnit(RBM.HiddenUnit.BINARY).build())
//                .layer(1,
//                        new RBM.Builder().nIn(500).nOut(250)
//                                .weightInit(WeightInit.XAVIER)
//                                .lossFunction(LossFunction.MCXENT)
//                                .visibleUnit(RBM.VisibleUnit.BINARY)
//                                .hiddenUnit(RBM.HiddenUnit.BINARY).build())
//                .layer(2,
//                        new RBM.Builder().nIn(250).nOut(200)
//                                .weightInit(WeightInit.XAVIER)
//                                .lossFunction(LossFunction.MCXENT)
//                                .visibleUnit(RBM.VisibleUnit.BINARY)
//                                .hiddenUnit(RBM.HiddenUnit.BINARY).build())
//                .layer(3,
//                        new OutputLayer.Builder(
//                                LossFunction.NEGATIVELOGLIKELIHOOD)
//                                .activation(Activation.SOFTMAX).nIn(200).nOut(outputNum)
//                                .build()).pretrain(true).backprop(false)
//                .build();
//
//        MultiLayerNetwork model = new MultiLayerNetwork(conf);
//        model.init();
//        model.setListeners(new ScoreIterationListener(listenerFreq)));
//
//        LOGGER.info("Train model....");
//        model.fit(mnistTrain); // achieves end to end pre-training
//
//        LOGGER.info("Evaluate model....");
//        Evaluation eval = new Evaluation(outputNum);
//
//        while (mnistTest.hasNext()) {
//            DataSet testMnist = mnistTest.next();
//            INDArray predict2 = model.output(testMnist.getFeatureMatrix());
//            eval.eval(testMnist.getLabels(), predict2);
//        }
//
//        LOGGER.info(eval.stats());
//        LOGGER.info("****************Example finished********************");
    }
}
