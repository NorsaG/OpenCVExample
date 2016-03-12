package org.firerate;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.ml.CvRTParams;
import org.opencv.ml.CvRTrees;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

/**
 * User: LazarevEvgeny
 * Date: 12.03.2016
 */
public class FireRateExample {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static final int[] OFFSETS = {0, 5, 10};

    public static void main(String[] args) throws IOException {
        List<String> csv = loadFile("fire_test.txt");
        CvRTrees tree;
        CvRTParams params = new CvRTParams();
        params.set_max_depth(2);

        Mat trainData = new Mat(csv.size() - 1, 3, CvType.CV_32F);
        Mat labels = new Mat(csv.size() - 1, 1, CvType.CV_32S);

        for (int offset : OFFSETS) {
            tree = new CvRTrees();
            for (int i = 1; i < csv.size(); i++) {
                String line = csv.get(i);
                String[] str = line.split(",");
                trainData.put(i - 1, 0, new float[]{Float.valueOf(str[0]), Float.valueOf(str[1]), Float.valueOf(str[2])});
                labels.put(i - 1, 0, new int[]{Integer.valueOf(str[3]) > offset ? 1 : 0});
            }
            tree.train(trainData, 1, labels, new Mat(), new Mat(), new Mat(), new Mat(), params);
            testModel(tree, offset);
        }
    }


    private static List<String> loadFile(String fileName) throws IOException {
        return Files.readAllLines(Paths.get(fileName));
    }

    private static void testModel(CvRTrees tree, int offset) {
        System.out.println("!!!!! Test model with count of fires more than " + offset + "!!!!!");
        testExample(tree, offset, 1, 28, 0.20f, 3);
        testExample(tree, offset, 2, 28, 0.90f, 3);

        testExample(tree, offset, 3, 30, 0.80f, 3);
        testExample(tree, offset, 4, 20, 0.80f, 3);

        testExample(tree, offset, 5, 25, 0.70f, 1);
        testExample(tree, offset, 6, 25, 0.70f, 6);
        System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    }


    private static void testExample(CvRTrees tree, int offset, int number, float temp, float humidity, float wind) {
        Mat ex = new Mat(1, 3, CvType.CV_32F);
        ex.put(0, 0, new float[]{temp, humidity, wind});

        System.out.println("Example " + number + "(count of fires -> " + offset + "): " + ex.dump());
        System.out.println(tree.predict(ex));
        System.out.println(tree.predict_prob(ex));
    }
}
