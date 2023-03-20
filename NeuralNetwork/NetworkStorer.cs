using System.Text.Json;

namespace NeuralNetwork
{
    public static class NetworkStorer
    {
        private static JsonSerializerOptions serializerOptions = new JsonSerializerOptions(JsonSerializerDefaults.General);

        public static void Save(Network network, string filePath)
        {
            using (Stream stream = File.OpenWrite(filePath))
            {
                byte[] serializedBytes = JsonSerializer.SerializeToUtf8Bytes(network, typeof(Network), serializerOptions);
                stream.Write(serializedBytes);
            }
        }

        public static Network Load(string filePath)
        {
            using (Stream stream = File.OpenRead(filePath))
            {
                Object? deserializedObject;
                try
                {
                    deserializedObject = JsonSerializer.Deserialize(stream, typeof(Network), serializerOptions);
                }
                catch (Exception exception)
                {
                    Console.WriteLine("Failed to deserialize XML file.");
                    throw;
                }

                Network? deserializedNetwork = deserializedObject as Network;
                if (deserializedNetwork == null)
                {
                    throw new Exception("Deserialized object was not a valid Network.");
                }
                else
                {
                    return deserializedNetwork;
                }
            }
        }

    }
}
