using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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

        public float[] GetNetworkOutput(float[] inputValues)
        {
            return Layers[0].GetOutputLayerValues(inputValues);
        }

        public float[][] GetAllNetworkValues(float[] inputValues)
        {
            return Layers[0].GetAllLayerValues(new float[][] { inputValues });
        }

        public void TrainNetwork()
        {
            int batchSize = 100;
            int epochs = 2;
            NetworkGradient gradientSum;
            for (int e = 0; e < epochs; e++)
            {
                for (int i = 0; i < batchSize; i++)
                {

                }
            }
        }

        public NetworkGradient BackpropagateNetwork(float[] inputValues, float[] desiredOutputValues)
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

            for (int n = 0; n < layerLength; n++)
            {
                outputError[n] = nodeValues[nodeValues.Length - 1][n] - desiredOutputValues[n];
                stackedDerivs[n] = CalculateSigmoidDerivative(nodeValues[nodeValues.Length - 1][n]) * 2 * outputError[n];
                layerBiasGradients[n] = stackedDerivs[n];
                for (int p = 0; p < nodeValues[nodeValues.Length - 2].Length; p++)
                {
                    layerWeightGradients[n, p] = stackedDerivs[n] * nodeValues[nodeValues.Length - 2][p];
                }
            }

            resultLayerGradients[resultLayerGradients.Length - 1] = new LayerGradient(layerBiasGradients, layerWeightGradients);

            for (int l = nodeValues.Length - 2; l > 0; l--)
            {
                layerBiasGradients = new float[nodeValues[l].Length];
                layerWeightGradients = new float[nodeValues[l].Length, nodeValues[l - 1].Length];
                float[] newDerivs = new float[nodeValues[l].Length];
                for (int n = 0; n < nodeValues[l].Length; n++)
                {
                    // for every node in prev. layer, multiply ^ by corresponding node in last layer to get weight gradients
                    // take sum of partial derivs to corresponding weights
                    // also this partial deriv = bias gradient, so it gets stored
                    for (int f = 0; f < nodeValues[l + 1].Length; f++) {
                        newDerivs[n] = newDerivs[n] + stackedDerivs[f] * Layers[l].Nodes[f].Weights[n];
                    }
                    newDerivs[n] *= CalculateSigmoidDerivative(nodeValues[l][n]);
                    layerBiasGradients[n] = newDerivs[n];
                    for (int p = 0; p < nodeValues[l - 1].Length; p++)
                    {
                        layerWeightGradients[n, p] = newDerivs[n] * nodeValues[l - 1][p];
                    }
                }
                stackedDerivs = newDerivs;
                resultLayerGradients[l - 1] = new LayerGradient(layerBiasGradients, layerWeightGradients);
            }

            resultNetworkGradient = new NetworkGradient(resultLayerGradients);
            return resultNetworkGradient;
        }

        public class NetworkGradient
        {
            public LayerGradient[] LayerGradients
            {
                get;
                set;
            }

            public NetworkGradient(LayerGradient[] layerGradients)
            {
                LayerGradients = layerGradients;
            }
        }

        public class LayerGradient
        {
            public float[] BiasGradients
            {
                get; 
                set;
            }

            public float[,] WeightGradients
            {
                get;
                set;
            }

            public LayerGradient(float[] biasGradients, float[,] weightGradients)
            {
                BiasGradients = biasGradients;
                WeightGradients = weightGradients;
            }
        }

        private static NetworkGradient SumNetworkGradients(NetworkGradient[] networkGradients)
        {
            int layers = networkGradients[0].LayerGradients.Length;
            LayerGradient[] layerGradientsSum = new LayerGradient[layers];
            
            for (int l = 0; l < layers; l++) 
            {
                int layerSize = networkGradients[0].LayerGradients[l].BiasGradients.Length;
                int prevLayerSize = networkGradients[0].LayerGradients[l].WeightGradients.GetUpperBound(2);
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
                randWeights[i] = (randomizer.NextSingle() - 0.5f) * 0.2f;
                //randBiases[i] = randomizer.NextSingle() - 0.5f;
            }

            var randBias = randomizer.NextSingle() * 0.2f;
            node.SetWeights(randWeights);
            node.Bias = randBias;
        }
    }
}
