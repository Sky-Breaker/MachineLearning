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
        public byte[,,] TrainingImages;
        public byte[] TrainingLabels;

        public int ImageSize;

        public TrainingDataReader(string trainingImagesFilePath, string trainingLabelsFilePath) {
            byte[] trainingImagesBytes = File.ReadAllBytes(trainingImagesFilePath);
            byte[] trainingLabelsBytes = File.ReadAllBytes(trainingLabelsFilePath);

            ImageSize = 28;

            TrainingImages = PrepareImageData(trainingImagesBytes);
            TrainingLabels = PrepareLabelData(trainingLabelsBytes);
        }

        private byte[,,] PrepareImageData(byte[] trainingImagesBytes)
        {
            byte[,,] trainingImageData = new byte[60000, ImageSize, ImageSize];
            for (int i = 16; i < trainingImagesBytes.Length; i++)
            {
                int currentImage = (i - 16) / (ImageSize * ImageSize);
                int pixelX = (i - 16) % ImageSize;
                int pixelY = ((i - 16) / ImageSize + 1) % ImageSize;
                trainingImageData[currentImage, pixelX, pixelY] = trainingImagesBytes[i];
            }
            return trainingImageData;
        }

        private byte[] PrepareLabelData(byte[] trainingLabelsBytes)
        {
            byte[] trainingLabelData = new byte[60000];
            for (int i = 8; i < trainingLabelsBytes.Length; i++)
            {
                trainingLabelData[i - 8] = trainingLabelsBytes[i];
            }
            return trainingLabelData;
        }

    }
}
