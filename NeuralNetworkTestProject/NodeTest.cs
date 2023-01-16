using NeuralNetwork;

namespace NeuralNetworkTestProject
{
    [TestClass]
    public class NodeTest
    {

        /// <summary>
        /// I'm expecting there to be two weights if I make a node of size 2.
        /// </summary>
        [TestMethod]
        public void ConstructingNodeShouldSetSize()
        {
            // Arrange 
            int sizeShouldBeTwo = 2;

            // Act
            var n = new Neuron(sizeShouldBeTwo);

            // Assert
            int expected = sizeShouldBeTwo;
            int actual = n.Weights.Length;
            Assert.AreEqual(expected, actual, "The size of the property should match");
        }


        [TestMethod]
        public void CalculateValueShouldThrowIfNoFloatInputs()
        {
            // Arrange
            var n = new Neuron(1);

            // Act
            // Assert
            float[] nullFloatArray = null;
            Assert.ThrowsException<ArgumentNullException>(() => { n.CalculateValue(nullFloatArray); });

        }

        /*
        [TestMethod]
        public void CalculateValueShouldThrowIfNoNodeInputs()
        {
            // Arrange
            var n = new Node(1);

            // Act
            // Assert
            Node[] nullFloatArray = null;
            Assert.ThrowsException<ArgumentNullException>(() => { n.CalculateValue(nullFloatArray); });

        }
        */
    }
}