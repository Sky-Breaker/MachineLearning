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
        private int Size;

        public ListOfData(byte[,] data)
        {
            Data = data;
            Size = data.GetLength(0);
        }

        public byte[] GetValuesAtIndex(int index)
        {
            byte[] values = new byte[Data.GetLength(1)];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = Data[index, i];
            }
            return values;
        }

        public int GetSize()
        {
            return Size;
        }
    }
}
