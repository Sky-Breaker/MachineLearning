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

        /*
        public void TrainNetwork(float[] inputValues, float[] desiredOutputValues)
        {
            float[][] nodeValues = GetAllNetworkValues(inputValues);

            float[,][] gradient = new float[nodeValues.Length,2][];

            float[] cost = new float[nodeValues[nodeValues.Length - 1].Length];
            for (int i = 0; i < cost.Length; i++)
            {
                cost[i] = 2 * (nodeValues[nodeValues.Length - 1][i] - desiredOutputValues[i]);
            }

            float[] partialDeriv = new float[nodeValues[nodeValues.Length - 1].Length];
            for (int l = nodeValues.Length - 1; l >= 0; l--)
            {
                for (int n = 0; n < nodeValues[l].Length; n++)
                {
                    partialDeriv[n] = CalculateSigmoidDerivative(nodeValues[l][n]) * cost[n];

                    gradient[l, 1][n] =
                }
            }
        }
        */

        // private float[][,] BackpropogateNetwork(float[] derivativesOfCostsToNextLayerActivation, float[][] networkValues)
        // {
        //     
        // }
        private float CalculateSigmoidDerivative(float x)
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
                randWeights[i] = randomizer.NextSingle() - 0.5f;
                //randBiases[i] = randomizer.NextSingle() - 0.5f;
            }

            var randBias = randomizer.NextSingle();
            node.SetWeights(randWeights);
            node.Bias = randBias;
        }
    }
}
