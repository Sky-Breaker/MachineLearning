using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;

namespace NeuralNetwork
{
    public class TrainingDataReader
    {
        public ListOfData TrainingImages;
        public ListOfData TrainingLabels;

        public ListOfData TestImages;
        public ListOfData TestLabels;

        public int ImageSize;
        public int LabelSize;

        public int[] trainingDataIndicies;

        public TrainingDataReader(string trainingImagesFilePath, string trainingLabelsFilePath, string testImagesFilePath, string testLabelsFilePath) {
            byte[] trainingImagesBytes = File.ReadAllBytes(trainingImagesFilePath);
            byte[] trainingLabelsBytes = File.ReadAllBytes(trainingLabelsFilePath);
            byte[] testImageBytes = File.ReadAllBytes(testImagesFilePath);
            byte[] testLabelsBytes = File.ReadAllBytes(testLabelsFilePath);

            ImageSize = 784;
            LabelSize = 10;

            TrainingImages = PrepareImageData(trainingImagesBytes, 60000);
            TrainingLabels = PrepareLabelData(trainingLabelsBytes, 60000);
            TestImages = PrepareImageData(testImageBytes, 10000);
            TestLabels = PrepareLabelData(testLabelsBytes, 10000);

            trainingDataIndicies = new int[TrainingImages.GetSize()];
            for (int i = 0; i < trainingDataIndicies.Length; i++)
            {
                trainingDataIndicies[i] = i;
            }
        }

        private ListOfData PrepareImageData(byte[] imagesBytes, int nOfImages)
        {
            byte[,] trainingImageData = new byte[nOfImages, ImageSize];
            for (int i = 16; i < imagesBytes.Length; i++)
            {
                int currentImage = (i - 16) / ImageSize;
                int pixelX = (i - 16) % ImageSize;
                //int pixelY = ((i - 16) / ImageSize + 1) % ImageSize;
                trainingImageData[currentImage, pixelX] = imagesBytes[i];
            }
            return new ListOfData(trainingImageData);
        }

        private ListOfData PrepareLabelData(byte[] labelsBytes, int nOfLabels)
        {
            byte[,] trainingLabelData = new byte[nOfLabels,10];
            for (int i = 8; i < labelsBytes.Length; i++)
            {
                byte[] label = new byte[10];
                for (int labelNumber = 0; labelNumber <= 9; labelNumber++)
                {
                    if (labelNumber == labelsBytes[i])
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

        /// <summary>
        /// Modern Fisher-Yates shuffling algorithm to get a random permutaion of the sequence of training data.
        /// Images will have the same index as their respective label.
        /// Source: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        /// </summary>
        public void ShuffleTrainingData()
        {
            Random randomizer = new Random();
            int trainingDataSize = trainingDataIndicies.Length;
            /*
            for (int i = 0; i < trainingDataSize - 1; i++)
            {
                int randomIndex = randomizer.Next(i, trainingDataSize);
                SwapData(i, randomIndex);
            }
            */
            Stopwatch timer = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < trainingDataSize - 1; i++)
            {
                int randomIndex = randomizer.Next(i, trainingDataSize);
                int tempValue = trainingDataIndicies[i];
                trainingDataIndicies[i] = trainingDataIndicies[randomIndex];
                trainingDataIndicies[randomIndex] = tempValue;
            }
            timer.Stop();
            var elapsedMs = timer.ElapsedMilliseconds;
            Console.Out.WriteLine("Shuffle elapsed Ms: " + elapsedMs);
        }

        /*
        private void SwapData(int index, int randomIndex)
        {
            byte[] tempImage = new byte[ImageSize];
            byte[] tempLabel = new byte[LabelSize];

            for (int i = 0; i < ImageSize; i++)
            {
                tempImage[i] = TrainingImages.GetValuesAtIndex(index)[i];
            }

            for (int i = 0; i < LabelSize; i++)
            {
                tempLabel[i] = TrainingLabels.GetValuesAtIndex(index)[i];
            }

            TrainingImages.SetValuesAtIndex(index, TrainingImages.GetValuesAtIndex(randomIndex));
            TrainingLabels.SetValuesAtIndex(index, TrainingLabels.GetValuesAtIndex(randomIndex));

            TrainingImages.SetValuesAtIndex(randomIndex, tempImage);
            TrainingLabels.SetValuesAtIndex(randomIndex, tempLabel);
        }
        */
    }
}
