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


public class FireRateExample {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static final int[] OFFSETS = {0, 5, 9}; // границы чисел. будем проверять и прогнозировать эти ситуации

    public static void main(String[] args) throws IOException {
        // загружаем данные
        List<String> csv = loadFile("fire_test.txt");
        CvRTrees tree;
        // настройка параметров. с этим можно долго и упорно играться, подгоняя так или иначе под модель
        CvRTParams params = new CvRTParams();
        params.set_max_depth(2);
//        params.set_nactive_vars(3);
//        params.set_min_sample_count(3);

        Mat trainData;
        Mat labels;
        // строим модель для ситуаций: кол-во пожаров > 0, > 5 и > 9
        for (int offset : OFFSETS) {
            tree = new CvRTrees();
            // создаем матрицу тренировочных данных размерности: х-1 чтобы отбросить названия колонок, 3 - количество критериев (CvType.CV_32F - тип данных)
            trainData = new Mat(csv.size() - 1, 3, CvType.CV_32F);
            // классификатор -> 0 или 1 (в нашем случае - количество пожаров больше определенного значения? проставляться будет позднее)
            labels = new Mat(csv.size() - 1, 1, CvType.CV_32S);
            // загружаем построчно данные и проставляем их в матрицы
            for (int i = 1; i < csv.size(); i++) {
                String line = csv.get(i);
                String[] str = line.split(",");
                trainData.put(i - 1, 0, new float[]{Float.valueOf(str[0]), Float.valueOf(str[1]), Float.valueOf(str[2])});
                labels.put(i - 1, 0, new int[]{Integer.valueOf(str[3]) > offset ? 1 : 0});
            }
            // тренируем и тестируем модель
            // 1-й параметр: входная модель без класса
            // 2-й параметр: тип входных данных (колонки(0) или строки(1))
            // 3-й параметр: значения для входных данных
//            tree.train(trainData, 1, labels);
            // 8-й параметр: параметры дерева
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

        testExample(tree, offset, 3, 35, 0.80f, 3);
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
