namespace NeuralNetwork
{
    public class Network
    {
        private Layer[] Layers;
        private ActivationFunction NodeOutputFunction;

        /// <param name="layerSizes">
        /// int[] whose elements are the length of each layer, from input to output. layerSizes[0] is the number of inputs of the network.
        /// The length of layerSizes must be at least 2 so the network can have a minimum of one layer.
        /// </param>
        /// <exception cref="ArgumentException">Thrown if the length of layerSizes is less than 2.</exception>
        public Network(int[] layerSizes, ActivationFunction activationFunction)
        {
            if (layerSizes.Length < 2)
            {
                throw new ArgumentException("Cannot create a network with less than 1 layer. layerSizes must be > 2. ", nameof(layerSizes));
            }

            NodeOutputFunction = activationFunction;

            // layerSizes contains the input size, so the amount of Layers in the network is one less than the length of layerSizes
            Layers = new Layer[layerSizes.Length - 1];

            // Reference to the next layer that each layer is given. Begins as null because the last layer should have a null reference.
            Layer nextLayer = null;
            // Because each layer needs a reference to the next layer, the last layer is created first followed by the previous layers.
            for (int l = Layers.Length - 1; l >= 0; l--)
            {
                /*
                // Since layerSizes contains the input size at index 0, the layer size will be at the next index in layerSizes from the corresponding Layer index l.
                var nodes = new Node[layerSizes[l + 1]];
                // For every Node in layer l:
                for (int n = 0; n < nodes.Length; n++)
                {
                    // Each node needs the size of the previous layer (or inputs), which is at index l for layer l because index 0 is the number of network inputs.
                    nodes[n] = new Node(layerSizes[l], NodeOutputFunction);
                    SetRandomStartingWeightsAndBiases(nodes[n]);
                }
                // Each layer is created with a reference to the next layer. The last layer's reference is null so recursion on the layers can end.
                Layers[l] = new Layer(nodes, nextLayer);
                */
                int layerSize = layerSizes[l + 1];
                int prevLayerSize = layerSizes[l];
                Layers[l] = new Layer(layerSize, prevLayerSize, nextLayer, NodeOutputFunction);
                nextLayer = Layers[l];
            }
        }

        public double[] GetNetworkOutput(byte[] inputValues)
        {
            // Convert byte array of inputs to floats for the recursive layer method
            double[] inputs = new double[inputValues.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = inputValues[i] / 255.0;
            }
            return Layers[0].GetOutputLayerValues(inputs);
        }

        public double[][] GetAllNetworkValues(byte[] inputValues)
        {
            // Convert byte array of inputs to floats for the recursive layer method
            double[] inputs = new double[inputValues.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = inputValues[i] / 255.0;
            }
            return Layers[0].GetAllLayerValues(new double[][] { inputs });
        }

        /// <summary>
        /// Trains the network on the given training data and labels.
        /// </summary>
        /// <param name="trainingImages">ListOfData containing the training images</param>
        /// <param name="trainingLabels">ListOfData containing the training labels</param>
        /// <param name="batchSize">Number of training examples to calculate a gradient for at one time.</param>
        /// <param name="learningRate">Constant multplied to the calculated gradient to adjust the network weights and biases by.</param>
        public void TrainNetwork(ListOfData trainingImages, ListOfData trainingLabels, int batchSize, double learningRate)
        {
            // Repeat batches until the end of the batch would exceed the last index of the training data.
            for (int batch = 0; (batch + 1) * batchSize < trainingImages.GetSize(); batch++)
            {
                int index = batch * batchSize;
                
                // Do the first backpropagation, and store the gradient and error
                BackpropagationResult backpropResult = BackpropagateNetwork(trainingImages.GetValuesAtIndex(index), trainingLabels.GetValuesAtIndex(index));
                NetworkGradient gradientSum = backpropResult.Gradient;
                double errorSum = backpropResult.Error;
                for (int i = 1; i < batchSize; i++)
                {
                    index++;
                    
                    // Do another backpropagation, and combine it with the last gradient to keep a sum.
                    backpropResult = BackpropagateNetwork(trainingImages.GetValuesAtIndex(index), trainingLabels.GetValuesAtIndex(index));
                    NetworkGradient newGradient = backpropResult.Gradient;
                    errorSum += backpropResult.Error;

                    NetworkGradient[] gradientsToCombine = new NetworkGradient[2];
                    gradientsToCombine[0] = gradientSum;
                    gradientsToCombine[1] = newGradient;

                    gradientSum = SumNetworkGradients(gradientsToCombine);
                }
                Console.Out.WriteLine("Error: " + errorSum / batchSize);
                AdjustNetworkValues(gradientSum, learningRate);
            }
        }

        /// <summary>
        /// Adjust every weight and bias in the network based on the given gradient and scaled by the learning rate.
        /// </summary>
        /// <param name="gradient">NetworkGradient containing calculated slopes of the cost of the network training data classifications with 
        /// respect to every network weight and bias.</param>
        /// <param name="learningRate">Constant scalar when multiplied with the negative gradient gives the adjustment for each weight and bias.</param>
        private void AdjustNetworkValues(NetworkGradient gradient, double learningRate)
        {
            for (int layer = 0; layer < Layers.Length; layer++)
            {
                for (int node = 0; node < Layers[layer].Nodes.Length; node++)
                {
                    // Because the gradient is a postive slope of cost to the respective weight or bias, the adjustment should be negative.
                    double biasAdjustmentValue = gradient.LayerGradients[layer].BiasGradients[node] * -1 * learningRate;
                    Layers[layer].Nodes[node].Bias += biasAdjustmentValue;

                    for (int weight = 0; weight < Layers[layer].Nodes[0].Weights.Length; weight++)
                    {
                        double weightAdjustmentValue = gradient.LayerGradients[layer].WeightGradients[node, weight] * -1 * learningRate;
                        Layers[layer].Nodes[node].Weights[weight] += weightAdjustmentValue;
                    }
                }
            }
        }

        /// <summary>
        /// Conducts backpropagation on the network given input values and the desired correct output values.
        /// </summary>
        /// <param name="inputValues">Training input values for the network.</param>
        /// <param name="desiredOutputValues">Correct desired network output given the inputs to calculate a gradient for.</param>
        /// <returns>BackpropagationResult containing the gradient of the network output cost with respect to the network weights and biases.</returns>
        public BackpropagationResult BackpropagateNetwork(byte[] inputValues, byte[] desiredOutputValues)
        {
            double[][] nodeValues = GetAllNetworkValues(inputValues);
            
            int layerLength = nodeValues[nodeValues.Length - 1].Length;
            NetworkGradient resultNetworkGradient;
            LayerGradient[] resultLayerGradients = new LayerGradient[nodeValues.Length - 1];

            double[] outputError = new double[layerLength];
            double[] stackedDerivs = new double[layerLength];

            // double arrays for bias and weight gradients in each layer
            double[] layerBiasGradients = new double[layerLength];
            double[,] layerWeightGradients = new double[layerLength, nodeValues[nodeValues.Length - 2].Length]; // [node, weight]

            double totalError = 0;
            // Calculate the last layer partial derivatives
            for (int node = 0; node < layerLength; node++)
            {
                outputError[node] = nodeValues[nodeValues.Length - 1][node] - desiredOutputValues[node];
                //stackedDerivs[node] = CalculateSigmoidDerivative(nodeValues[nodeValues.Length - 1][node]) * 2 * outputError[node];
                stackedDerivs[node] = NodeOutputFunction.DerivativeAt(nodeValues[nodeValues.Length - 1][node]) * 2 * outputError[node];
                layerBiasGradients[node] = stackedDerivs[node];
                for (int prevNode = 0; prevNode < nodeValues[nodeValues.Length - 2].Length; prevNode++)
                {
                    layerWeightGradients[node, prevNode] = stackedDerivs[node] * nodeValues[nodeValues.Length - 2][prevNode];
                }
                totalError += Math.Pow(outputError[node], 2);
            }

            resultLayerGradients[resultLayerGradients.Length - 1] = new LayerGradient(layerBiasGradients, layerWeightGradients);

            // Calculate the partial derivatives for each preceding layer
            for (int layer = nodeValues.Length - 2; layer > 0; layer--)
            {
                layerBiasGradients = new double[nodeValues[layer].Length];
                layerWeightGradients = new double[nodeValues[layer].Length, nodeValues[layer - 1].Length];
                double[] newDerivs = new double[nodeValues[layer].Length];
                for (int node = 0; node < nodeValues[layer].Length; node++)
                {
                    // The partial derivative of the nodes in this layer with respect to the partial derivatives of the next layer is
                    // influenced by every node in the next layer.
                    for (int nextNode = 0; nextNode < nodeValues[layer + 1].Length; nextNode++) {
                        newDerivs[node] = newDerivs[node] + stackedDerivs[nextNode] * Layers[layer].Nodes[nextNode].Weights[node];
                    }
                    //newDerivs[node] *= CalculateSigmoidDerivative(nodeValues[layer][node]);
                    newDerivs[node] *= NodeOutputFunction.DerivativeAt(nodeValues[layer][node]);
                    layerBiasGradients[node] = newDerivs[node];
                    for (int prevNode = 0; prevNode < nodeValues[layer - 1].Length; prevNode++)
                    {
                        layerWeightGradients[node, prevNode] = newDerivs[node] * nodeValues[layer - 1][prevNode];
                    }
                }
                stackedDerivs = newDerivs;
                resultLayerGradients[layer - 1] = new LayerGradient(layerBiasGradients, layerWeightGradients);
            }

            resultNetworkGradient = new NetworkGradient(resultLayerGradients);
            BackpropagationResult result = new BackpropagationResult(resultNetworkGradient, totalError);
            return result;
        }

        /// <summary>
        /// Returns a NetworkGradient containing the sum of the individual gradients in all of the given network gradients.
        /// </summary>
        /// <param name="networkGradients">Array of network gradients to sum.</param>
        /// <returns>A network gradient whose layer's individual weights and biases are the sum of the respective weight or bias
        /// in every given network gradient.</returns>
        private static NetworkGradient SumNetworkGradients(NetworkGradient[] networkGradients)
        {
            int layers = networkGradients[0].LayerGradients.Length;
            LayerGradient[] layerGradientsSum = new LayerGradient[layers];

            for (int l = 0; l < layers; l++) 
            {
                int layerSize = networkGradients[0].LayerGradients[l].BiasGradients.Length;
                int prevLayerSize = networkGradients[0].LayerGradients[l].WeightGradients.GetUpperBound(1) + 1;

                double[] biasGradientSum = new double[layerSize];
                double[,] weightGradientSum = new double[layerSize, prevLayerSize];

                for (int g = 0; g < networkGradients.Length; g++) 
                {
                    for (int b = 0; b < layerSize; b++) {
                        biasGradientSum[b] += networkGradients[g].LayerGradients[l].BiasGradients[b];
                        for (int w = 0; w < prevLayerSize; w++)
                        {
                            weightGradientSum[b, w] += networkGradients[g].LayerGradients[l].WeightGradients[b, w];
                        }
                    }
                }
                layerGradientsSum[l] = new LayerGradient(biasGradientSum, weightGradientSum);
            }
            NetworkGradient networkGradientSum = new NetworkGradient(layerGradientsSum);
            return networkGradientSum;
        }

        /*
        private static double CalculateSigmoidDerivative(double x)
        {
            double eToX = Math.Exp(x);
            return eToX / Math.Pow(eToX + 1, 2);
        }
        */

        private void SetRandomStartingWeightsAndBiases(Node node)
        {
            var nodeInputSize = node.Weights.Length;
            var randomizer = new Random();

            var randWeights = new double[nodeInputSize];
            // var randBiases = new double[nodeInputSize];

            for (int i = 0; i < nodeInputSize; i++)
            {
                randWeights[i] = randomizer.NextDouble() - 0.5;
                //randBiases[i] = randomizer.NextSingle() - 0.5f;
            }

            var randBias = randomizer.NextDouble() - 0.5;
            node.SetWeights(randWeights);
            node.Bias = randBias;
        }
    }
}
