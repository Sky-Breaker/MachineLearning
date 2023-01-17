namespace NeuralNetwork
{
    public class Network
    {
        private Layer[] Layers;

        /// <param name="layerSizes">
        /// int[] whose elements are the length of each layer, from input to output. layerSizes[0] is the number of inputs of the network.
        /// The length of layerSizes must be at least 2 so the network can have a minimum of one layer.
        /// </param>
        /// <exception cref="ArgumentException">Thrown if the length of layerSizes is less than 2.</exception>
        public Network(int[] layerSizes)
        {
            if (layerSizes.Length < 2)
            {
                throw new ArgumentException("Cannot create a network with less than 1 layer. layerSizes must be > 2. ", nameof(layerSizes));
            }

            // layerSizes contains the input size, so the amount of Layers in the network is one less than the length of layerSizes
            Layers = new Layer[layerSizes.Length - 1];

            // Reference to the next layer that each layer is given. Begins as null because the last layer should have a null reference.
            Layer nextLayer = null;
            // Because each layer needs a reference to the next layer, the last layer is created first followed by the previous layers.
            for (int l = Layers.Length - 1; l >= 0; l--)
            {
                // Since layerSizes contains the input size at index 0, the layer size will be at the next index in layerSizes from the corresponding Layer index l.
                var nodes = new Node[layerSizes[l + 1]];
                // For every Node in layer l:
                for (int n = 0; n < nodes.Length; n++)
                {
                    // Each node needs the size of the previous layer (or inputs), which is at index l for layer l because index 0 is the number of network inputs.
                    nodes[n] = new Node(layerSizes[l]);
                    SetRandomStartingWeightsAndBiases(nodes[n]);
                }
                // Each layer is created with a reference to the next layer. The last layer's reference is null so recursion on the layers can end.
                Layers[l] = new Layer(nodes, nextLayer);
                nextLayer = Layers[l];
            }
        }

        public float[] GetNetworkOutput(byte[] inputValues)
        {
            float[] inputs = new float[inputValues.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = inputValues[i];
            }
            return Layers[0].GetOutputLayerValues(inputs);
        }

        public float[][] GetAllNetworkValues(byte[] inputValues)
        {
            float[] inputs = new float[inputValues.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = inputValues[i];
            }
            return Layers[0].GetAllLayerValues(new float[][] { inputs });
        }

        public void TrainNetwork(ListOfData trainingImages, ListOfData trainingLabels, int batchSize, float learningRate)
        {
            for (int batch = 0; (batch + 1) * batchSize < trainingImages.GetSize(); batch++)
            {
                int index = batch * batchSize;

                NetworkGradient gradientSum = BackpropagateNetwork(trainingImages.GetValuesAtIndex(index), trainingLabels.GetValuesAtIndex(index));
                for (int i = 1; i < batchSize; i++)
                {
                    index++;

                    NetworkGradient result = BackpropagateNetwork(trainingImages.GetValuesAtIndex(index), trainingLabels.GetValuesAtIndex(index));

                    NetworkGradient[] gradientsToCombine = new NetworkGradient[2];
                    gradientsToCombine[0] = gradientSum;
                    gradientsToCombine[1] = result;

                    gradientSum = SumNetworkGradients(gradientsToCombine);
                }
                AdjustNetworkValues(gradientSum, learningRate);
            }
        }

        private void AdjustNetworkValues(NetworkGradient gradient, float learningRate)
        {
            for (int layer = 0; layer < Layers.Length; layer++)
            {
                for (int node = 0; node < Layers[layer].Nodes.Length; node++)
                {
                    float biasAdjustmentValue = gradient.LayerGradients[layer].BiasGradients[node] * -1 * learningRate;
                    Layers[layer].Nodes[node].Bias += biasAdjustmentValue;

                    for (int weight = 0; weight < Layers[layer].Nodes[0].Weights.Length; weight++)
                    {
                        float weightAdjustmentValue = gradient.LayerGradients[layer].WeightGradients[node, weight] * -1 * learningRate;
                        Layers[layer].Nodes[node].Weights[weight] += weightAdjustmentValue;
                    }
                }
            }
        }

        public NetworkGradient BackpropagateNetwork(byte[] inputValues, byte[] desiredOutputValues)
        {
            float[][] nodeValues = GetAllNetworkValues(inputValues);
            
            int layerLength = nodeValues[nodeValues.Length - 1].Length;
            NetworkGradient resultNetworkGradient;
            LayerGradient[] resultLayerGradients = new LayerGradient[nodeValues.Length - 1];

            float[] outputError = new float[layerLength];
            float[] stackedDerivs = new float[layerLength];

            // float arrays for bias and weight gradients in each layer
            float[] layerBiasGradients = new float[layerLength];
            float[,] layerWeightGradients = new float[layerLength, nodeValues[nodeValues.Length - 2].Length]; // [node, weight]

            string errorAmounts = "Error: ";

            for (int node = 0; node < layerLength; node++)
            {
                outputError[node] = nodeValues[nodeValues.Length - 1][node] - desiredOutputValues[node];
                stackedDerivs[node] = CalculateSigmoidDerivative(nodeValues[nodeValues.Length - 1][node]) * 2 * outputError[node];
                layerBiasGradients[node] = stackedDerivs[node];
                for (int prevNode = 0; prevNode < nodeValues[nodeValues.Length - 2].Length; prevNode++)
                {
                    layerWeightGradients[node, prevNode] = stackedDerivs[node] * nodeValues[nodeValues.Length - 2][prevNode];
                }
                errorAmounts += Math.Pow(outputError[node], 2) + ", ";
            }
            Console.Out.WriteLine(errorAmounts);

            resultLayerGradients[resultLayerGradients.Length - 1] = new LayerGradient(layerBiasGradients, layerWeightGradients);

            for (int layer = nodeValues.Length - 2; layer > 0; layer--)
            {
                layerBiasGradients = new float[nodeValues[layer].Length];
                layerWeightGradients = new float[nodeValues[layer].Length, nodeValues[layer - 1].Length];
                float[] newDerivs = new float[nodeValues[layer].Length];
                for (int node = 0; node < nodeValues[layer].Length; node++)
                {
                    // for every node in prev. layer, multiply ^ by corresponding node in last layer to get weight gradients
                    // take sum of partial derivs to corresponding weights
                    // also this partial deriv = bias gradient, so it gets stored
                    for (int nextNode = 0; nextNode < nodeValues[layer + 1].Length; nextNode++) {
                        newDerivs[node] = newDerivs[node] + stackedDerivs[nextNode] * Layers[layer].Nodes[nextNode].Weights[node];
                    }
                    newDerivs[node] *= CalculateSigmoidDerivative(nodeValues[layer][node]);
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
            return resultNetworkGradient;
        }

        private static NetworkGradient SumNetworkGradients(NetworkGradient[] networkGradients)
        {
            int layers = networkGradients[0].LayerGradients.Length;
            LayerGradient[] layerGradientsSum = new LayerGradient[layers];
            
            for (int l = 0; l < layers; l++) 
            {
                int layerSize = networkGradients[0].LayerGradients[l].BiasGradients.Length;
                int prevLayerSize = networkGradients[0].LayerGradients[l].WeightGradients.GetUpperBound(1) + 1;

                float[] biasGradientSum = new float[layerSize];
                float[,] weightGradientSum = new float[layerSize, prevLayerSize];

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

        private static float CalculateSigmoidDerivative(float x)
        {
            float eToX = MathF.Exp(x);
            return eToX / MathF.Pow(eToX + 1, 2);
        }

        private void SetRandomStartingWeightsAndBiases(Node node)
        {
            var nodeInputSize = node.Weights.Length;
            var randomizer = new Random();

            var randWeights = new float[nodeInputSize];
            // var randBiases = new float[nodeInputSize];

            for (int i = 0; i < nodeInputSize; i++)
            {
                randWeights[i] = (randomizer.NextSingle() - 0.5f) * 1f;
                //randBiases[i] = randomizer.NextSingle() - 0.5f;
            }

            var randBias = randomizer.NextSingle() * 1f;
            node.SetWeights(randWeights);
            node.Bias = randBias;
        }
    }
}
