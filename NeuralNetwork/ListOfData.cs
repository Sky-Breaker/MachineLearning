using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class ListOfData
    {
        private byte[,] Data;

        public ListOfData(byte[,] data)
        {
            Data = data;
        }

        public byte[] GetValuesAtIndex(int index)
        {
            byte[] values = new byte[Data.GetLength(2)];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = Data[1, i];
            }
            return values;
        }
    }
}
