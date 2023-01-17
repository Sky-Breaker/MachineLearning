using System.ComponentModel;
using System.Xml.Linq;

namespace NeuralNetwork
{
    internal class Program
    {
        static void Main(string[] args)
        {
            
            String trainingImagesFilePath = "C:\\Users\\ffpil\\OneDrive\\Documents\\Projects\\MNISTDataset\\UnzippedTrainingData\\train-images.idx3-ubyte";
            String trainingLabelsFilePath = "C:\\Users\\ffpil\\OneDrive\\Documents\\Projects\\MNISTDataset\\UnzippedTrainingData\\train-labels.idx1-ubyte";
            String testImagesFilePath = "C:\\Users\\ffpil\\OneDrive\\Documents\\Projects\\MNISTDataset\\UnzippedTrainingData\\t10k-images.idx3-ubyte";
            String testLabelsFilePath = "C:\\Users\\ffpil\\OneDrive\\Documents\\Projects\\MNISTDataset\\UnzippedTrainingData\\t10k-labels.idx1-ubyte";

            TrainingDataReader trainingData = new TrainingDataReader(trainingImagesFilePath, trainingLabelsFilePath, testImagesFilePath, testLabelsFilePath);

            var networkSize = new int[4] { 784, 200, 100, 10 };

            var network = new Network(networkSize);

            int epochs = 4;
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                network.TrainNetwork(trainingData.TrainingImages, trainingData.TrainingLabels, 400, 0.005f);//*MathF.Pow(0.4f,epoch)
                trainingData.ShuffleTrainingData();
            }

            float correctlyLabeled = 0;
            for (int t = 0; t < trainingData.TestImages.GetSize(); t++)
            {
                float[] networkOutput = network.GetNetworkOutput(trainingData.TestImages.GetValuesAtIndex(t));
                float maxOutput = -1;
                int indexOfMaxOutput = 0;
                for (int index = 0; index < networkOutput.Length; index++)
                {
                    if (maxOutput < networkOutput[index])
                    {
                        maxOutput = networkOutput[index];
                        indexOfMaxOutput = index;
                    }
                }
                int indexOfTestLabel = 0;
                for (int index = 0; index < 10; index++)
                {
                    if (trainingData.TestLabels.GetValuesAtIndex(t)[index] == 1)
                    {
                        indexOfTestLabel = index;
                    }
                }
                //Console.Out.WriteLine("Classified as: " + indexOfMaxOutput + " Actual: " + indexOfTestLabel);
                if (indexOfTestLabel == indexOfMaxOutput)
                {
                    correctlyLabeled += 1;
                }
            }

            Console.Out.WriteLine("Percent correct: " + (correctlyLabeled / trainingData.TestImages.GetSize()) * 100);



            /*
            while (true)
            {
                Console.Out.WriteLine("Enter a test image number:");
                string testImageInput = Console.In.ReadLine();
                int testImageNumber = 0;
                try 
                {
                    int.TryParse(testImageInput,out testImageNumber);
                }
                catch
                {
                    Console.Out.WriteLine("Incorrect format for test image number provided.");
                }
                float[] networkOutput = network.GetNetworkOutput(trainingData.TestImages.GetValuesAtIndex(testImageNumber));

                string outputText = "Network output: ";
                for (int i = 0; i < networkOutput.Length; i++)
                {
                    outputText += i + "-" + networkOutput[i] + ", ";
                }

                string correctOutputText = "Expected output: ";
                for (int i = 0; i < networkOutput.Length; i++)
                {
                    correctOutputText += i + "-" + trainingData.TestLabels.GetValuesAtIndex(testImageNumber)[i] + ", ";
                }

                Console.WriteLine(outputText);
                Console.WriteLine(correctOutputText);
            }
            */


            /*
            var randomizer = new Random();

            var networkSize = new int[4] { 784, 20, 16, 10 };

            var testNetwork = new Network(networkSize);

            var inputs = new float[networkSize[0]];
            for (int i = 0; i < networkSize[0]; i++)
            {
                float r = 32 * MathF.Pow(randomizer.NextSingle() + 0.15f,15);
                if (r < 0)
                {
                    r = 0;
                }
                // inputs[i] = randomizer.NextSingle() - 0.5f;
                inputs[i] = r;
            }

            var time = DateTime.Now;
            //var outputs = testNetwork.GetNetworkOutput(inputs);
            var outputs = testNetwork.GetAllNetworkValues(inputs);
            var timeSpan = DateTime.Now - time;

            var inputString = "Inputs: ";
            var outputString = "Outputs: ";


            for (int i = 0; i < outputs.Length; i++)
            {
                outputString += "\nLayer: " + i + "\n";
                for (int j = 0; j < outputs[i].Length; j++)
                {
                    outputString += outputs[i][j] + ", ";
                }
            }


            Console.Out.WriteLine(inputString);
            Console.Out.WriteLine(outputString);
            Console.Out.WriteLine("Elapsed Time: " + timeSpan.TotalMilliseconds + "ms");

            String testOutput = "";
            float[] backpropTestImage = new float[28 * 28];
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++) {
                    Boolean PixelIsOn = trainingData.TrainingImages.GetValuesAtIndex[18, i * 28 + j] > 0;
                    if (PixelIsOn)
                    {
                        testOutput += "*";
                    }
                    else
                    {
                        testOutput += " ";
                    }
                    backpropTestImage[28*i+j] = trainingData.TrainingImages[18, i * 28 + j];
                }
                testOutput += "\n";
            }

            Console.Write(testOutput);
            

            float[] expectedValues = { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
            Network.NetworkGradient testg = testNetwork.BackpropagateNetwork(backpropTestImage, expectedValues);


            var randomizer = new Random();
            var inputSize = 6;

            var firstNodes = new Node[4];
            firstNodes[0] = new Node(inputSize);
            firstNodes[1] = new Node(inputSize);
            firstNodes[2] = new Node(inputSize);
            firstNodes[3] = new Node(inputSize);

            var secondNodes = new Node[3];
            secondNodes[0] = new Node(firstNodes.Length);
            secondNodes[1] = new Node(firstNodes.Length);
            secondNodes[2] = new Node(firstNodes.Length);

            var thirdNodes = new Node[3];
            thirdNodes[0] = new Node(secondNodes.Length);
            thirdNodes[1] = new Node(secondNodes.Length);
            thirdNodes[2] = new Node(secondNodes.Length);
            
            foreach (Node node in firstNodes)
            {
                var randWeights = new float[node.Weights.Length];
                var randBiases = new float[node.Weights.Length];

                for (int i = 0; i < node.Weights.Length; i++)
                {
                    randWeights[i] = randomizer.NextSingle();
                    randBiases[i] = randomizer.NextSingle();
                }

                node.Weights = randWeights;
                node.Biases = randBiases;
            }

            foreach (Node node in secondNodes)
            {
                var randWeights = new float[node.Weights.Length];
                var randBiases = new float[node.Weights.Length];

                for (int i = 0; i < node.Weights.Length; i++)
                {
                    randWeights[i] = randomizer.NextSingle();
                    randBiases[i] = randomizer.NextSingle();
                }

                node.Weights = randWeights;
                node.Biases = randBiases;
            }

            foreach (Node node in thirdNodes)
            {
                var randWeights = new float[node.Weights.Length];
                var randBiases = new float[node.Weights.Length];

                for (int i = 0; i < node.Weights.Length; i++)
                {
                    randWeights[i] = randomizer.NextSingle();
                    randBiases[i] = randomizer.NextSingle();
                }

                node.Weights = randWeights;
                node.Biases = randBiases;
            }

            var extInputs = new float[inputSize];

            for (int i = 0; i < inputSize; i++)
            {
                extInputs[i] = randomizer.NextSingle();
            }

            var outputLayer = new Layer(thirdNodes, null);
            var secondLayer = new Layer(secondNodes, outputLayer);
            var firstLayer = new Layer(firstNodes, secondLayer);

            var time = DateTime.Now;

            var outputs = firstLayer.CalculateAllLayers(extInputs);

            var timeSpan = DateTime.Now - time;

            var inputString = "Inputs: ";
            var outputString = "Outputs: ";

            for (int i = 0; i < extInputs.Length; i++)
            {
                inputString += extInputs[i] + ", ";
            }

            for (int i = 0; i < outputs.Length; i++)
            {
                outputString += outputs[i] + ", ";
            }

            Console.Out.WriteLine(inputString);
            Console.Out.WriteLine(outputString);
            Console.Out.WriteLine("Elapsed Time: " + timeSpan.TotalMilliseconds + "ms");
            */
        }
    }
}