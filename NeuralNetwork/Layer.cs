using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>
    /// A group of nodes in the neural network that are all at the same depth. It can contain a reference to the next layer, which will be null for
    /// the output layer. It can call CalculateValue on every node it contains and return these values, and can also recursively get the values from
    /// the output layer. 
    /// </summary>
    public class Layer
    {
        public Neuron[] Nodes
        {
            get;
        }

        private Layer NextLayer;

        public Layer(Neuron[] nodeList, Layer nextLayer) {
            Nodes = nodeList;
            NextLayer = nextLayer;
        }

        /// <summary>
        /// Gets the value of every node in the layer.
        /// </summary>
        /// <param name="inputs">Input values to be given to each node.</param>
        /// <returns>A float array containing the values of each node in the layer, obtained by calling CalculateValue() on them.</returns>
        public float[] CalculateLayer(float[] inputs)
        {
            float[] nodeValues = new float[Nodes.Length];

            for (int i = 0; i < Nodes.Length; i++)
            {
                nodeValues[i] = Nodes[i].CalculateValue(inputs);
            }

            return nodeValues;
        }

        /// <summary>
        /// Calculates the values of the nodes in every layer and gives back the output when called from the first layer.
        /// </summary>
        /// <param name="inputs">External input values for the entire network.</param>
        /// <returns>The output values of the network.</returns>
        /// <remarks>Each node needs inputs from the last layer (or the external inputs for nodes in the input layer) when calculating
        /// it's value. Also, this recursive method needs to call itself on the next layer. Each layer contains a reference to the next
        /// layer so it can make the next layer calculate its value, and then the output layer which has a null reference can end the
        /// recursion and send the output values back to the caller. The parameter of inputs--which comes from the previous layer--is
        /// given to the nodes in the layer, and then the values of the nodes in the layer is passed on in the recursive call.</remarks>
        public float[] GetOutputLayerValues(float[] inputs)
        {
            if (NextLayer == null)
            {
                // Console.WriteLine("Recursion ended.");
                return CalculateLayer(inputs);
            }
            else
            {
                return NextLayer.GetOutputLayerValues(CalculateLayer(inputs));
            }
        }

        public float[][] GetAllLayerValues(float[][] inputs)
        {
            float[] lastLayerValues = inputs[inputs.Length - 1];

            float[][] groupedLayers = new float[inputs.Length + 1][];

            for (int i = 0; i < inputs.Length; i++)
            {
                groupedLayers[i] = inputs[i];
            }

            groupedLayers[groupedLayers.Length - 1] = CalculateLayer(lastLayerValues);

            if (NextLayer == null)
            {
                return groupedLayers;
            }
            else
            {   
                return NextLayer.GetAllLayerValues(groupedLayers);
            }
        }
    }
}
