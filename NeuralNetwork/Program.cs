using System.Xml.Linq;

namespace NeuralNetwork
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Console.WriteLine("Hello. My name is JARVIS.");
            var randomizer = new Random();

            var networkSize = new int[4] { 10, 4, 3, 3};

            var testNetwork = new Network(networkSize);

            var inputs = new float[networkSize[0]];
            for (int i = 0; i < networkSize[0]; i++)
            {
                inputs[i] = randomizer.NextSingle() - 0.5f;
            }

            var time = DateTime.Now;
            //var outputs = testNetwork.GetNetworkOutput(inputs);
            var outputs = testNetwork.GetAllNetworkValues(inputs);
            var timeSpan = DateTime.Now - time;

            var inputString = "Inputs: ";
            var outputString = "Outputs: ";

            /*
            for (int i = 0; i < networkSize[0]; i++)
            {
                inputString += inputs[i] + ", ";
            }
            */

            /*
            for (int i = 0; i < outputs.Length; i++)
            {
                outputString += outputs[i] + ", ";
            }
            */

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


            String trainingImagesFilePath = "C:\\Users\\ffpil\\OneDrive\\Documents\\Projects\\MNISTDataset\\UnzippedTrainingData\\train-images.idx3-ubyte";
            String trainingLabelsFilePath = "C:\\Users\\ffpil\\OneDrive\\Documents\\Projects\\MNISTDataset\\UnzippedTrainingData\\train-labels.idx1-ubyte";

            TrainingDataReader trainingData = new TrainingDataReader(trainingImagesFilePath, trainingLabelsFilePath);
            String testOutput = "";
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++) {
                    Boolean PixelIsOn = trainingData.TrainingImages[18, j, i] > 0;
                    if (PixelIsOn)
                    {
                        testOutput += "*";
                    }
                    else
                    {
                        testOutput += " ";
                    }
                }
                testOutput += "\n";
            }

            Console.Write(testOutput);

            /*
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