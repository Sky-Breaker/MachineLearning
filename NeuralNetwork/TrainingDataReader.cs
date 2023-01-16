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

        public ListOfData TestImages;
        public ListOfData TestLabels;

        public int ImageSize;

        public TrainingDataReader(string trainingImagesFilePath, string trainingLabelsFilePath, string testImagesFilePath, string testLabelsFilePath) {
            byte[] trainingImagesBytes = File.ReadAllBytes(trainingImagesFilePath);
            byte[] trainingLabelsBytes = File.ReadAllBytes(trainingLabelsFilePath);
            byte[] testImageBytes = File.ReadAllBytes(testImagesFilePath);
            byte[] testLabelsBytes = File.ReadAllBytes(testLabelsFilePath);

            ImageSize = 28;

            TrainingImages = PrepareImageData(trainingImagesBytes, 60000);
            TrainingLabels = PrepareLabelData(trainingLabelsBytes, 60000);
            TestImages = PrepareImageData(testImageBytes, 10000);
            TestLabels = PrepareLabelData(testLabelsBytes, 10000);
        }

        private ListOfData PrepareImageData(byte[] trainingImagesBytes, int nOfImages)
        {
            byte[,] trainingImageData = new byte[nOfImages, ImageSize * ImageSize];
            for (int i = 16; i < trainingImagesBytes.Length; i++)
            {
                int currentImage = (i - 16) / (ImageSize * ImageSize);
                int pixelX = (i - 16) % (ImageSize * ImageSize);
                //int pixelY = ((i - 16) / ImageSize + 1) % ImageSize;
                trainingImageData[currentImage, pixelX] = trainingImagesBytes[i];
            }
            return new ListOfData(trainingImageData);
        }

        private ListOfData PrepareLabelData(byte[] trainingLabelsBytes, int nOfLabels)
        {
            byte[,] trainingLabelData = new byte[nOfLabels,10];
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
