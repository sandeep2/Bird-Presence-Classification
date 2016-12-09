import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.*;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import java.io.*;
import java.util.*;


public class project3 {
    public static void main(String[] args) throws IllegalArgumentException, IOException, ClassNotFoundException, InterruptedException {
        Configuration conf = new Configuration();
        try {
            Job job = Job.getInstance(conf, "PageRank");
            job.setJarByClass(PageRank.class);
            job.setMapperClass(InitialMapper.class);
            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(Text.class);
            job.setReducerClass(InitialReducer.class);
            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(NullWritable.class);
            FileInputFormat.addInputPath(job, new Path("/home/sandeep/Downloads/labeled_50000.csv"));
            FileOutputFormat.setOutputPath(job, new Path("/home/sandeep/Downloads/output"));
            job.waitForCompletion(true);

            job = Job.getInstance(conf, "PageRank");
            job.setJarByClass(PageRank.class);
            job.setMapperClass(classifyMapper.class);
            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(Text.class);
            job.setReducerClass(classifyReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(NullWritable.class);
            FileInputFormat.addInputPath(job, new Path("/home/sandeep/Downloads/1234.csv"));
            FileOutputFormat.setOutputPath(job, new Path("/home/sandeep/Downloads/output1"));
            job.waitForCompletion(true);


        } catch (Exception e) {
            e.printStackTrace();
        }


    }

    public static class InitialMapper extends Mapper<Object, Text, IntWritable, Text> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            int temp = 1 + (int) (Math.random() * ((10 - 1) + 1));
            String[] line = value.toString().split(",");
            if (!line[4].equals("YEAR")) {
                context.write(new IntWritable(temp), value);
            }
        }
    }


    public static class InitialReducer extends Reducer<IntWritable, Text, IntWritable, NullWritable> {
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            ArrayList<Attribute> attributes = new ArrayList<Attribute>();
            Instances data;
            attributes.add(new Attribute("latitude"));
            attributes.add(new Attribute("longitude"));
            attributes.add(new Attribute("Year"));
            attributes.add(new Attribute("Month"));
            attributes.add(new Attribute("Day"));
            attributes.add(new Attribute("Time"));
            attributes.add(new Attribute("Effort_Hrs"));
            attributes.add(new Attribute("Effort_Distance_Km"));
            attributes.add(new Attribute("Effort_Area_Ha"));
            attributes.add(new Attribute("Number_Observers"));
            attributes.add(new Attribute("Primary_Checklist_Flag"));
            attributes.add(new Attribute("NLCD2001_FS_C11_7500_PLAND"));
            attributes.add(new Attribute("NLCD2001_FS_C12_7500_PLAND"));
            attributes.add(new Attribute("NLCD2001_FS_C21_7500_PLAND"));
            attributes.add(new Attribute("NLCD2001_FS_C22_7500_PLAND"));
            attributes.add(new Attribute("NLCD2001_FS_C23_7500_PLAND"));

            attributes.add(new Attribute("NLCD2001_FS_C24_7500_PLAND"));
            attributes.add(new Attribute("NLCD2006_FS_C82_7500_PLAND"));
            attributes.add(new Attribute("NLCD2006_FS_C90_7500_PLAND"));
            attributes.add(new Attribute("NLCD2006_FS_C95_7500_PLAND"));

            attributes.add(new Attribute("NLCD2011_FS_C11_7500_PLAND"));
            attributes.add(new Attribute("NLCD2011_FS_C12_7500_PLAND"));
            attributes.add(new Attribute("NLCD2011_FS_C21_7500_PLAND"));
            attributes.add(new Attribute("NLCD2011_FS_C22_7500_PLAND"));
            attributes.add(new Attribute("NLCD2011_FS_C23_7500_PLAND"));
            attributes.add(new Attribute("NLCD2011_FS_C24_7500_PLAND"));
            attributes.add(new Attribute("NLCD2011_FS_C31_7500_PLAND"));
            attributes.add(new Attribute("NLCD2011_FS_C41_7500_PLAND"));


            attributes.add(new Attribute("Bird"));
            data = new Instances(new String("Data"), attributes, 0);
            data.setClassIndex(data.numAttributes() - 1);
            int j = 0;


            for (Text each : values) {
                String[] line = each.toString().split(",");
                int[] temp = new int[]{2, 3, 4, 5, 6, 7, 12, 13, 14, 16, 18, 968, 969, 970, 971, 972, 973, 997, 998, 999, 1000,
                        1001, 1002, 1003, 1004, 1005, 1006, 1007, 26};
                double[] temp2 = new double[29];
                if (j != 0) {
                    int i = 0;
                    for (int k=0; k<temp.length;k++) {
                        try {
                            if(line[temp[k]].equals("X") || line[temp[k]].equals("?")){
                                temp2[i]=Double.parseDouble("0.0");
                            }
                            else{
                                if(k == 28){
                                    Double classify = Double.parseDouble(line[temp[k]]);
                                    if(classify == 0){
                                        temp2[i] = Double.parseDouble(line[temp[k]]);}
                                    else{
                                        temp2[i] = Double.parseDouble("1");
                                    }
                                }
                                else{
                                    temp2[i] = Double.parseDouble(line[temp[k]]);
                                }
                            }} catch (Exception e) {
                            e.printStackTrace();
                        }
                        i++;
                    }
                }
                DenseInstance di = new DenseInstance(1.0, temp2);
                data.add(di);
                j++;
            }


            try {
                ReplaceMissingValues rmv = new ReplaceMissingValues();
                rmv.setInputFormat(data);
                data = Filter.useFilter(data, rmv);
                String[] options = new String[2];
                options[0] = "-R";
                options[1] = "last";

                NumericToNominal nn = new NumericToNominal();
                nn.setOptions(options);
                nn.setInputFormat(data);
                data = Filter.useFilter(data, nn);


                SMOTE sm = new SMOTE();
                sm.setPercentage(140);
                sm.setNearestNeighbors(7);
                sm.setInputFormat(data);

//                AttributeSelection filter = new AttributeSelection();
//                CfsSubsetEval eval = new CfsSubsetEval();
//                GreedyStepwise search = new GreedyStepwise();
//                search.setSearchBackwards(true);
//                filter.setEvaluator(eval);
//                filter.setSearch(search);
//                filter.setInputFormat(data);
//                // generate new data
//                data = Filter.useFilter(data, filter);
//


                if (key.get() == 1 || key.get() == 2 || key.get()==3) {
                    Evaluation eval = new Evaluation(data);
                    Random rand = new Random(1);
                    int folds = 10;
                    ArrayList<Double> accuracy_list = new ArrayList<Double>();
                    RandomTree randomTree = new RandomTree();
                    ArrayList<RandomTree> svmArray = new ArrayList<RandomTree>();


                    for (int n = 0; n < folds; n++) {
                        Instances train = data.trainCV(folds, n);
                        Instances test = data.testCV(folds, n);
                        //randomTree.setOptions(weka.core.Utils.splitOptions("-C 0.1 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
                        randomTree.buildClassifier(train);
                        eval.evaluateModel(randomTree, test);
                        Double temp = eval.numTruePositives(0)+eval.numTrueNegatives(0);
                        svmArray.add(randomTree);
                        accuracy_list.add(temp);
                    }
                    randomTree = svmArray.get(accuracy_list.indexOf(Collections.max(accuracy_list)));
                    weka.core.SerializationHelper.write("/home/sandeep/Desktop/rt.model"+"-"+key.get(),randomTree);
                }



                if (key.get() == 4 || key.get() == 5 || key.get() == 6) {
                    Evaluation eval = new Evaluation(data);
                    Random rand = new Random(1);
                    int folds = 10;
                    ArrayList<Double> accuracy_list = new ArrayList<Double>();
                    J48 j48 = new J48();
                    ArrayList<J48> j48Array = new ArrayList<J48>();
                    for (int n = 0; n < folds; n++) {
                        Instances train = data.trainCV(folds, n);
                        Instances test = data.testCV(folds, n);
                        j48.buildClassifier(train);
                        eval.evaluateModel(j48, test);
                        Double temp = eval.numTruePositives(0)+eval.numTrueNegatives(0);
                        j48Array.add(j48);
                        accuracy_list.add(temp);
                    }
                    j48 = j48Array.get(accuracy_list.indexOf(Collections.max(accuracy_list)));
                    weka.core.SerializationHelper.write("/home/sandeep/Desktop/j48.model"+"-"+key.get(),j48);
                }

//                if (key.get() == 3) {
//                    Evaluation eval = new Evaluation(data);
//                    Random rand = new Random(1);
//                    int folds = 10;
//                    ArrayList<Double> accuracy_list = new ArrayList<Double>();
//                    LMT randomForest = new LMT();
//                    ArrayList<LMT> j48Array = new ArrayList<LMT>();
//                    for (int n = 0; n < folds; n++) {
//                        Instances train = data.trainCV(folds, n);
//                        Instances test = data.testCV(folds, n);
//                        randomForest.buildClassifier(train);
//                        eval.evaluateModel(randomForest, test);
//                        Double temp = eval.numTruePositives(0)+eval.numTrueNegatives(0);
//                        j48Array.add(randomForest);
//                        accuracy_list.add(temp);
//                    }
//                    randomForest = j48Array.get(accuracy_list.indexOf(Collections.max(accuracy_list)));
//                    weka.core.SerializationHelper.write("/home/sandeep/Desktop/rf.model",randomForest);
//                }

                if (key.get() == 7 || key.get() == 8 || key.get() == 9 || key.get() == 10) {
                    Evaluation eval = new Evaluation(data);
                    int folds = 10;
                    ArrayList<Double> accuracy_list = new ArrayList<Double>();
                    REPTree repTree = new REPTree();
                    ArrayList<REPTree> dsArray = new ArrayList<REPTree>();
                    for (int n = 0; n < folds; n++) {
                        Instances train = data.trainCV(folds, n);
                        Instances test = data.testCV(folds, n);
                        repTree.buildClassifier(train);
                        eval.evaluateModel(repTree, test);
                        Double temp = eval.numTruePositives(0)+eval.numTrueNegatives(0);
                        dsArray.add(repTree);
                        accuracy_list.add(temp);
                    }
                    repTree = dsArray.get(accuracy_list.indexOf(Collections.max(accuracy_list)));
                    weka.core.SerializationHelper.write("/home/sandeep/Desktop/rp.model"+"-"+key.get(),repTree);
                }

            } catch (Exception e) {
                e.printStackTrace();
            }

            context.write(key, NullWritable.get());
        }

    }

    public static class classifyMapper extends Mapper<Object, Text, IntWritable, Text> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            int temp = 1 + (int) (Math.random() * ((10 - 1) + 1));
            String[] line = value.toString().split(",");
            if (!line[4].equals("YEAR")) {
                context.write(new IntWritable(1), value);
            }
        }
    }

    public static class classifyReducer extends Reducer<IntWritable, Text, Text, NullWritable> {
        Instances data;
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            ArrayList<Attribute> attributes = new ArrayList<Attribute>();
            attributes.add(new Attribute("latitude"));
            attributes.add(new Attribute("longitude"));
            attributes.add(new Attribute("Year"));
            attributes.add(new Attribute("Month"));
            attributes.add(new Attribute("Day"));
            attributes.add(new Attribute("Time"));
            attributes.add(new Attribute("Effort_Hrs"));
            attributes.add(new Attribute("Effort_Distance_Km"));
            attributes.add(new Attribute("Effort_Area_Ha"));
            attributes.add(new Attribute("Number_Observers"));
            attributes.add(new Attribute("Primary_Checklist_Flag"));
            attributes.add(new Attribute("NLCD2001_FS_C11_7500_PLAND"));
            attributes.add(new Attribute("NLCD2001_FS_C12_7500_PLAND"));
            attributes.add(new Attribute("NLCD2001_FS_C21_7500_PLAND"));
            attributes.add(new Attribute("NLCD2001_FS_C22_7500_PLAND"));
            attributes.add(new Attribute("NLCD2001_FS_C23_7500_PLAND"));
            attributes.add(new Attribute("NLCD2001_FS_C24_7500_PLAND"));
            attributes.add(new Attribute("NLCD2006_FS_C82_7500_PLAND"));
            attributes.add(new Attribute("NLCD2006_FS_C90_7500_PLAND"));
            attributes.add(new Attribute("NLCD2006_FS_C95_7500_PLAND"));
            attributes.add(new Attribute("NLCD2011_FS_C11_7500_PLAND"));
            attributes.add(new Attribute("NLCD2011_FS_C12_7500_PLAND"));
            attributes.add(new Attribute("NLCD2011_FS_C21_7500_PLAND"));
            attributes.add(new Attribute("NLCD2011_FS_C22_7500_PLAND"));
            attributes.add(new Attribute("NLCD2011_FS_C23_7500_PLAND"));
            attributes.add(new Attribute("NLCD2011_FS_C24_7500_PLAND"));
            attributes.add(new Attribute("NLCD2011_FS_C31_7500_PLAND"));
            attributes.add(new Attribute("NLCD2011_FS_C41_7500_PLAND"));
            attributes.add(new Attribute("Bird"));

            data = new Instances(new String("Data"), attributes, 0);
            data.setClassIndex(data.numAttributes() - 1);
            int j = 0;

            for (Text each : values) {
                String[] line = each.toString().split(",");
                context.write(new Text(line[0]),NullWritable.get());
                int[] temp = new int[]{2, 3, 4, 5, 6, 7, 12, 13, 14, 16, 18, 968, 969, 970, 971, 972, 973, 997, 998, 999, 1000,
                        1001, 1002, 1003, 1004, 1005, 1006, 1007, 26};
                double[] temp2 = new double[29];
                if (j != 0) {
                    int i = 0;
                    for (int k=0; k<temp.length;k++) {
//                        String decimalPattern = "([0-9]*)\\.([0-9]*)";
//                        boolean match = Pattern.matches(decimalPattern, line[k]);
                        try {
                            if(line[temp[k]].equals("X")|| line[temp[k]].equals("?")){
                                temp2[i]=Double.parseDouble("0.0");
                            }
                            else{
                                if(k == 28){
                                    Double classify = Double.parseDouble(line[temp[k]]);
                                    if(classify == 0){
                                        temp2[i] = Double.parseDouble(line[temp[k]]);}
                                    else{
                                        temp2[i] = Double.parseDouble("1");
                                    }
                                }
                                else{
                                    temp2[i] = Double.parseDouble(line[temp[k]]);
                                }
                            }} catch (Exception e) {
                            e.printStackTrace();
                            //System.out.println(line[attribute]+"-------------------------------------------------");
                        }
                        i++;
                    }
                }
                DenseInstance di = new DenseInstance(1.0, temp2);
                data.add(di);
                j++;
            }


            try {
                ReplaceMissingValues rmv = new ReplaceMissingValues();
                rmv.setInputFormat(data);
                data = Filter.useFilter(data, rmv);
                String[] options = new String[2];
                options[0] = "-R";
                options[1] = "last";

                NumericToNominal nn = new NumericToNominal();
                nn.setOptions(options);
                nn.setInputFormat(data);
                data = Filter.useFilter(data, nn);


                Evaluation eval = new Evaluation(data);
                Classifier j48_1= (Classifier) weka.core.SerializationHelper.read("/home/sandeep/Desktop/j48.model-4");
                eval.evaluateModel(j48_1, data);
                System.out.println(eval.toMatrixString());

                Evaluation eval_j48 = new Evaluation(data);
                Classifier j48_2 = (Classifier) weka.core.SerializationHelper.read("/home/sandeep/Desktop/j48.model-5");
                eval_j48.evaluateModel(j48_2, data);
                System.out.println(eval_j48.toMatrixString());

                Evaluation eval_j48_1 = new Evaluation(data);
                Classifier j48_3 = (Classifier) weka.core.SerializationHelper.read("/home/sandeep/Desktop/j48.model-6");
                eval_j48_1.evaluateModel(j48_3, data);
                System.out.println(eval_j48_1.toMatrixString());

                Classifier randomtree_1 = (Classifier) weka.core.SerializationHelper.read("/home/sandeep/Desktop/rt.model-1");
                Evaluation eval_rt_1 = new Evaluation(data);
                eval_rt_1.evaluateModel(randomtree_1, data);
                System.out.println(eval_rt_1.toMatrixString());

                Classifier randomtree_2 = (Classifier) weka.core.SerializationHelper.read("/home/sandeep/Desktop/rt.model-2");
                Evaluation eval_rt_2 = new Evaluation(data);
                eval_rt_2.evaluateModel(randomtree_2, data);
                System.out.println(eval_rt_2.toMatrixString());

                Classifier randomtree_3 = (Classifier) weka.core.SerializationHelper.read("/home/sandeep/Desktop/rt.model-3");
                Evaluation eval_rt_3 = new Evaluation(data);
                eval_rt_3.evaluateModel(randomtree_3, data);
                System.out.println(eval_rt_3.toMatrixString());


//                Classifier randomforest = (Classifier) weka.core.SerializationHelper.read("/home/sandeep/Desktop/rf.model");
//                Evaluation eval_rf = new Evaluation(data);
//                eval_rf.evaluateModel(randomforest, data);
//                System.out.println(eval_rf.toMatrixString());

                Classifier reptree_1 = (Classifier) weka.core.SerializationHelper.read("/home/sandeep/Desktop/rp.model-7");
                Evaluation eval_rp_1 = new Evaluation(data);
                eval_rp_1.evaluateModel(reptree_1, data);
                System.out.println(eval_rp_1.toMatrixString());

                Classifier reptree_2 = (Classifier) weka.core.SerializationHelper.read("/home/sandeep/Desktop/rp.model-8");
                Evaluation eval_rp_2 = new Evaluation(data);
                eval_rp_2.evaluateModel(reptree_2, data);
                System.out.println(eval_rp_2.toMatrixString());

                Classifier reptree_3 = (Classifier) weka.core.SerializationHelper.read("/home/sandeep/Desktop/rp.model-9");
                Evaluation eval_rp_3 = new Evaluation(data);
                eval_rp_3.evaluateModel(reptree_3, data);
                System.out.println(eval_rp_3.toMatrixString());

                Classifier reptree_4 = (Classifier) weka.core.SerializationHelper.read("/home/sandeep/Desktop/rp.model-10");
                Evaluation eval_rp_4 = new Evaluation(data);
                eval_rp_4.evaluateModel(reptree_4, data);
                System.out.println(eval_rp_4.toMatrixString());

                Double a,b,c,d,e,f,g,h,ii,jj;
                for(int i=0; i< data.numInstances(); i++){

                    try {
                        int zero = 0;
                        int one = 0;
                        a = randomtree_1.classifyInstance(data.instance(i));
                        if (a == 0){
                            zero++;
                        }else{one++;}
                        c = randomtree_2.classifyInstance(data.instance(i));
                        if (c == 0){
                            zero++;
                        }else{one++;}
                        d = randomtree_3.classifyInstance(data.instance(i));
                        if (d == 0){
                            zero++;
                        }else{one++;}
                        g = reptree_4.classifyInstance(data.instance(i));
                        if(g ==0){
                            zero++;
                        }else{one++;}
                        b = reptree_1.classifyInstance(data.instance(i));
                        if(b ==0){
                            zero++;
                        }else{one++;}
                        e = reptree_2.classifyInstance(data.instance(i));
                        if(e ==0){
                            zero++;
                        }else{one++;}
                        f = reptree_3.classifyInstance(data.instance(i));
                        if(f ==0){
                            zero++;
                        }else{one++;}
                        h = j48_1.classifyInstance(data.instance(i));
                        if(h ==0){
                            zero++;
                        }else{one++;}
                        ii = j48_2.classifyInstance(data.instance(i));
                        if(ii ==0){
                            zero++;
                        }else{one++;}
                        jj = j48_3.classifyInstance(data.instance(i));
                        if(jj ==0){
                            zero++;
                        }else{one++;}


                        if(zero >= one){
                            context.write(new Text("0"),NullWritable.get());
                        }
                        else{
                            context.write(new Text("1"),NullWritable.get());
                        }
                    }
                    catch (Exception ex){
                        ex.printStackTrace();
                        // System.out.println(data.instance(i)+"---------------------------------------------------------");
                    }

                }

            }catch (Exception e){
                e.printStackTrace();
            }
        }


    }
}
