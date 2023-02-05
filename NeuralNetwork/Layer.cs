using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace NeuralNetwork
{
    /// <summary>
    /// A group of nodes in the neural network that are all at the same depth. It can contain a reference to the next layer, which will be null for
    /// the output layer. It can call CalculateValue on every node it contains and return these values, and can also recursively get the values from
    /// the output layer. 
    /// </summary>
    public class Layer
    {
        public Node[] Nodes
        {
            get;
        }

        private Layer NextLayer;

        private ActivationFunction NodeActivationFunction;


        public Layer(int layerSize, int prevLayerSize, Layer nextLayer, ActivationFunction activationFunction)
        {
            Nodes = new Node[layerSize];
            for (int n = 0; n < layerSize; n++)
            {
                Node node = new Node(prevLayerSize, activationFunction);
                node.SetRandomStartingWeightsAndBiases();
                Nodes[n] = node;
            }
            NextLayer = nextLayer;
            NodeActivationFunction = activationFunction;
        }

        /// <summary>
        /// Gets the value of every node in the layer.
        /// </summary>
        /// <param name="inputs">Input values to be given to each node.</param>
        /// <returns>A double array containing the values of each node in the layer, obtained by calling CalculateValue() on them.</returns>
        public double[] CalculateLayer(double[] inputs)
        {
            double[] nodeValues = new double[Nodes.Length];

            for (int i = 0; i < Nodes.Length; i++)
            {
                nodeValues[i] = Nodes[i].CalculateValue(inputs);
            }

            return nodeValues;
        }

        /// <summary>
        /// Calculates the values of the nodes in every layer and gives back the output when called from the first layer.
        /// </summary>
        /// <param name="inputs">Input values to be given to the nodes in the layer. Initial recursive call should pass
        /// the external input values for the entire network.</param>
        /// <returns>The output values of the network after recursively forward propagating through it.</returns>
        public double[] GetOutputLayerValues(double[] inputs)
        {
            if (NextLayer == null)
            {
                return CalculateLayer(inputs);
            }
            else
            {
                return NextLayer.GetOutputLayerValues(CalculateLayer(inputs));
            }
        }

        /// <summary>
        /// Calculates the values of the nodes in every layer and gives back every node value when called from the first layer.
        /// </summary>
        /// <param name="inputs">Input values to be given to the nodes in the layer, along with previous layers' values. Initial 
        /// recursive call should pass the external input values for the entire network in a 2d jagged double array of size
        /// [1][*number of inputs to the network*].</param>
        /// <returns>The values of every node in the network after recursively forward propagating through it.</returns>
        public double[][] GetAllLayerValues(double[][] inputs)
        {
            // The values to calculate this layer on should be the values of the last layer.
            double[] lastLayerValues = inputs[inputs.Length - 1];

            // A jagged 2d array to store the values of the nodes in this layer, and the ones in previous layers.
            double[][] groupedLayers = new double[inputs.Length + 1][];

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
