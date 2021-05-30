import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class IrisApp {
    public static void main(String[] args) throws Exception {
        int batchSize=1; // parcourrir le fichier ligne par ligne
        int outputSize=3;
        int classIndex=4;
        double learninRate=0.001;
        int inputSize=4;
        int numHiddenNodes=10;
        int nEpochs=100;
        System.out.println("Creation du modele !********************************************************");

        String[] labels={"Iris-setosa","Iris-versicolor","Iris-virginica"};
        MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(learninRate))
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(inputSize).nOut(numHiddenNodes).activation(Activation.SIGMOID).build())
                .layer(1,new OutputLayer.Builder()
                        .nIn(numHiddenNodes).nOut(outputSize)
                        .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .activation(Activation.SOFTMAX).build())
                .build();
        MultiLayerNetwork model=new MultiLayerNetwork(configuration);
        model.init();
        System.out.println(configuration.toJson());

        /*
        * demarrage du serveur pour le monitoring des resultats(processus) d'apprentissage
        * */

        UIServer uiServer = UIServer.getInstance();
        InMemoryStatsStorage inMemoryStatsStorage = new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);

        model.setListeners(new StatsListener(inMemoryStatsStorage));

        System.out.println("entrainnement du model !!! ********************************************************");


        File fileTrain = new ClassPathResource("iris-train.csv").getFile();
        RecordReader recordReaderTrain = new CSVRecordReader();
        recordReaderTrain.initialize(new FileSplit(fileTrain));
        DataSetIterator dataSetIteratorTrain = new RecordReaderDataSetIterator(
                recordReaderTrain,
                batchSize,
                classIndex,
                outputSize
                );

        /*while(dataSetIteratorTrain.hasNext()){
            System.out.println("----------------------------------------------------------");
            DataSet dataSet = dataSetIteratorTrain.next();
            System.out.println(dataSet.getFeatures());
            System.out.println(dataSet.getLabels());

        }*/

        /*
        * apprentissage du model avec fit qui va se repeter nPoch fois
        * */
        for(int i=0;i<nEpochs;i++){
            model.fit(dataSetIteratorTrain);
        }

        /*
        * evauation du model avec le fichier test
        * */

        System.out.println("evaluation du model ********************************************************");
        File fileTest = new ClassPathResource("irisTest.csv").getFile();
        RecordReader recordReaderTest = new CSVRecordReader();
        recordReaderTest.initialize(new FileSplit(fileTest));
        DataSetIterator dataSetIteratorTest = new RecordReaderDataSetIterator(
                recordReaderTest,
                batchSize,
                classIndex,
                outputSize
                );

        Evaluation evaluation = new Evaluation();
        while(dataSetIteratorTest.hasNext()){
            System.out.println("----------------------------------------------------------");
            DataSet dataSetTest = dataSetIteratorTest.next();
            NDArray featuresTest = (NDArray) dataSetTest.getFeatures();
            NDArray labelsTest = (NDArray) dataSetTest.getLabels();
            NDArray predicted = (NDArray) model.output(featuresTest);
            evaluation.eval(predicted, labelsTest);
        }
        System.out.println(evaluation.stats());
        ModelSerializer.writeModel(model, "irisModel.zip", true);


    }
}
