using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>
    /// Individual part of the neural network that contains a value. This value is calculated either from a sum of stored weights and a bias
    /// applied to each node in the previous layer or the inputs for nodes in the input layer.
    /// </summary>
    public class Node
    {
        public float[] Weights
        {
            get;
            set;
        }

        public float Bias
        {
            get;
            set;
        }

        public Node(int nOfInputs)
        {
            Weights = new float[nOfInputs];
            Bias = 0;
        }

        /// <summary>
        /// Calculates and returns the node value using the node's weights and bias along with each input value.
        /// </summary>
        /// <param name="inputs">Input values for the node that will be used to calcuate this node's value. Must be initialized.</param>
        /// <returns>The calculated node value, which is the sigmoid function applied to the sum of the weights times the corresponding input added to the bias:
        /// sigm(sum[n = 1 -> nOfNodes](weight_n*input_n) + bias)</returns>
        /// <exception cref="ArgumentNullException">Thrown if inputs is not provided</exception>
        public float CalculateValue(float[] inputs)
        {
            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs), "Inputs must be provided to calculate value. ");
            }

            float total = Bias; // Include bias in total
            // Iterate through every input and add weight * input to total
            for (int i = 0; i < inputs.Length; i++)
            {
                total += Weights[i] * inputs[i];
            }

            //Logistic sigmoid function, equal to 1 / (1 + e^-x)
            return 1 / (1 + MathF.Exp(-total));
        }

        /// <summary>
        /// Sets the weights of the node to the new ones given.
        /// </summary>
        /// <param name="newWeights">Array containing the new weights to be assigned to this node. Size must match the original size of weights.</param>
        /// <exception cref="ArgumentException">Thrown if the size of new weights doesn't match the amount of weights this node has.</exception>
        public void SetWeights(float[] newWeights)
        {
            if (newWeights.Length != Weights.Length)
            {
                throw new ArgumentException("Size of new weights doesn't match the amount of weights this node has. ", nameof(newWeights));
            }

            Weights = newWeights;
        }

        /// <summary>
        /// Sets the biases of the node to the new ones given.
        /// </summary>
        /// <param name="newBias">Array containing the new biases to be assigned to this node. Size must match the original size of biases.</param>
        public void SetBias(float newBias)
        {
            Bias = newBias;
        }

    }
}
