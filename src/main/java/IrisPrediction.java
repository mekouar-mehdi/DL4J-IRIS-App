import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

public class IrisPrediction {

    public static void main(String[] args) throws Exception {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("irisModel.zip"));

        String[] labels={"Iris-setosa","Iris-versicolor","Iris-virginica"};


        /*
         * prediction d'un exemple
         * */


        /*
                {4.9,3.0,1.4,0.2},
                {6.7,3.1,4.4,1.4},
                {5.6,3.0,4.5,1.5},
                {6.0,3.0,4.8,1.8},
                {6.9,3.1,5.4,2.1}
        * */

        System.out.println("prediction de 6 exemples des fleurs sous forme de tableau *********************************");

        NDArray inputData = (NDArray) Nd4j.create(new double[][]{
                {5.1,3.5,1.4,0.2},
                {4.9,3.0,1.4,0.2},
                {6.7,3.1,4.4,1.4},
                {5.6,3.0,4.5,1.5},
                {6.0,3.0,4.8,1.8},
                {6.9,3.1,5.4,2.1}
        });

        INDArray output = model.output(inputData);
        int[] classes = output.argMax(1).toIntVector(); //par ligne de la matrice

        System.out.println("prediction => " + output);

        for(int i=0;i<classes.length;i++)

        {
            System.out.println("classe => " + classes[i] + " nom => " + labels[classes[i]]);
        }
    }
}
