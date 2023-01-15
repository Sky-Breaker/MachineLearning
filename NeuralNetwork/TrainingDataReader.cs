using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace NeuralNetwork
{
    public class TrainingDataReader
    {
        public ListOfData TrainingImages;
        public ListOfData TrainingLabels;

        public int ImageSize;

        public TrainingDataReader(string trainingImagesFilePath, string trainingLabelsFilePath) {
            byte[] trainingImagesBytes = File.ReadAllBytes(trainingImagesFilePath);
            byte[] trainingLabelsBytes = File.ReadAllBytes(trainingLabelsFilePath);

            ImageSize = 28;

            TrainingImages = PrepareImageData(trainingImagesBytes);
            TrainingLabels = PrepareLabelData(trainingLabelsBytes);
        }

        private ListOfData PrepareImageData(byte[] trainingImagesBytes)
        {
            byte[,] trainingImageData = new byte[60000, ImageSize * ImageSize];
            for (int i = 16; i < trainingImagesBytes.Length; i++)
            {
                int currentImage = (i - 16) / (ImageSize * ImageSize);
                int pixelX = (i - 16) % (ImageSize * ImageSize);
                //int pixelY = ((i - 16) / ImageSize + 1) % ImageSize;
                trainingImageData[currentImage, pixelX] = trainingImagesBytes[i];
            }
            return new ListOfData(trainingImageData);
        }

        private ListOfData PrepareLabelData(byte[] trainingLabelsBytes)
        {
            byte[,] trainingLabelData = new byte[60000,10];
            for (int i = 8; i < trainingLabelsBytes.Length; i++)
            {
                byte[] label = new byte[10];
                for (int labelNumber = 0; labelNumber <= 9; labelNumber++)
                {
                    if (labelNumber == trainingLabelsBytes[i])
                    {
                        label[labelNumber] = 1;
                    }
                    else
                    {
                        label[labelNumber] = 0;
                    }
                    trainingLabelData[i - 8, labelNumber] = label[labelNumber];
                }
            }
            return new ListOfData(trainingLabelData);
        }

    }
}
