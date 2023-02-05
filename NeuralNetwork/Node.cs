namespace NeuralNetwork
{
    /// <summary>
    /// Individual part of the neural network that contains a value. This value is calculated either from a sum of stored weights and a bias
    /// applied to each node in the previous layer or the inputs for nodes in the input layer.
    /// </summary>
    public class Node
    {
        public double[] Weights;

        public double Bias;

        private ActivationFunction NodeOutputFunction;

        public Node(int nOfInputs, ActivationFunction activationFunction)
        {
            Weights = new double[nOfInputs];
            Bias = 0;

            NodeOutputFunction = activationFunction;
        }

        /// <summary>
        /// Calculates and returns the node value using the node's weights and bias along with each input value.
        /// </summary>
        /// <param name="inputs">Input values for the node that will be used to calcuate this node's value. Must be initialized.</param>
        /// <returns>The calculated node value, which is the sigmoid function applied to the sum of the weights times the corresponding input added to the bias:
        /// sigm(sum[n = 1 -> nOfNodes](weight_n*input_n) + bias)</returns>
        /// <exception cref="ArgumentNullException">Thrown if inputs is not provided</exception>
        public double CalculateValue(double[] inputs)
        {
            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs), "Inputs must be provided to calculate value. ");
            }

            double total = Bias; // Include bias in total
            // Iterate through every input and add weight * input to total
            for (int i = 0; i < inputs.Length; i++)
            {
                total += Weights[i] * inputs[i];
            }

            // Apply the logistic sigmoid function to total input, where sig(x) = 1 / (1 + e^-x)
            // return 1 / (1 + Math.Exp(-total));
            return NodeOutputFunction.ValueAt(total);
        }

        /// <summary>
        /// Sets the weights of the node to the new ones given.
        /// </summary>
        /// <param name="newWeights">Array containing the new weights to be assigned to this node. Size must match the original size of weights.</param>
        /// <exception cref="ArgumentException">Thrown if the size of new weights doesn't match the amount of weights this node has.</exception>
        public void SetWeights(double[] newWeights)
        {
            if (newWeights.Length != Weights.Length)
            {
                throw new ArgumentException("Size of new weights doesn't match the amount of weights this node has. ", nameof(newWeights));
            }

            Weights = newWeights;
        }

        public void SetRandomStartingWeightsAndBiases()
        {
            var nodeInputSize = Weights.Length;
            var randomizer = new Random();

            var randWeights = new double[nodeInputSize];

            for (int i = 0; i < nodeInputSize; i++)
            {
                randWeights[i] = randomizer.NextDouble() - 0.5;
            }

            var randBias = randomizer.NextDouble() - 0.5;
            Weights = randWeights;
            Bias = randBias;
        }
    }
}
