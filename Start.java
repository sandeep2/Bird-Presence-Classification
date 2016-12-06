import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Created by sandeep on 12/5/16.
 */
public class project2 {
    public static void main(String[] args) throws IllegalArgumentException, IOException, ClassNotFoundException, InterruptedException {
        Configuration conf = new Configuration();
        try {
            Job job = Job.getInstance(conf, "PageRank");
            job.setJarByClass(PageRank.class);
            job.setMapperClass(PageRankSort.class);
            job.addCacheFile(new java.net.URI("/home/sandeep/Downloads/temp2.csv"));
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
            FileInputFormat.addInputPath(job, new Path("/home/sandeep/Downloads/train.csv"));
            FileOutputFormat.setOutputPath(job, new Path("/home/sandeep/Downloads/output"));
            job.waitForCompletion(true);
        }catch (Exception e){
            e.printStackTrace();
        }

    }

    public static class PageRankSort extends Mapper<Object, Text, Text, Text> {
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        Instances data;
        Instances testset;
        int j;
        protected void setup(Context context) throws IOException, InterruptedException{
            java.net.URI[] netURI = context.getCacheFiles();
            Path mapped = new Path(netURI[0]);
            CSVLoader loader = new CSVLoader();
            ArffSaver saver = new ArffSaver();
            loader.setSource(new File(mapped.toString()));
            Instances data2 = loader.getDataSet();
            saver.setInstances(data2);
            saver.setFile(new File("/home/sandeep/Downloads/mynewfile3.arff"));
            saver.writeBatch();
            try {
                ConverterUtils.DataSource source1 = new ConverterUtils.DataSource("/home/sandeep/Downloads/mynewfile3.arff");
                testset = source1.getDataSet();
            }catch (Exception e){
                e.printStackTrace();
            }
            attributes.add(new Attribute("latitude"));
            attributes.add(new Attribute("longitude"));
            attributes.add(new Attribute("Year"));
            attributes.add(new Attribute("Month"));
            attributes.add(new Attribute("Day"));
//            FastVector birdpresent = new FastVector<>();
//            birdpresent.add(1.0);
//            birdpresent.add(0.0);
            attributes.add(new Attribute("Bird"));
            data = new Instances(new String("Data"),attributes,0);
            data.setClassIndex(data.numAttributes()-1);
            j = 0;
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // Emiting top 100 values from every mapper buy storing every element in treemap.
            // Removing the first element if there are more than 100 elements in treemap.
            String[] line = value.toString().split(",");
            int[] temp = new int[]{3,4,5,6,7,27};
            double[] values = new double[6];
            if(j != 0){
                int i =0;
                for(int each:temp){
                    String decimalPattern = "([0-9]*)\\.([0-9]*)";
                    boolean match = Pattern.matches(decimalPattern, line[each]);
                    values[i] = Double.parseDouble(line[each]);
                    i++;
                }
            }
            DenseInstance di = new DenseInstance(1.0,values);
            data.add(di);
            j++;
        }
        protected void cleanup(Context context) throws IOException,
                InterruptedException {

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
                data = Filter.useFilter(data,nn);

                NaiveBayes nb = new NaiveBayes();
                data.setClassIndex(data.numAttributes()-1);
                nb.buildClassifier(data);

                testset.setClassIndex(testset.numAttributes()-1);
                for(int i=0;i< testset.numInstances();i++){
                    System.out.println("nb----"+nb.classifyInstance(testset.instance(i)));
                    context.write(new Text("1---"),new Text(""+nb.classifyInstance(testset.instance(i))));
                }
            }catch (Exception e){
                e.printStackTrace();
            }
        }
    }

}

