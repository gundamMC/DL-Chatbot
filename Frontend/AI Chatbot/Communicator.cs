using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Net.Sockets;

namespace AI_Chatbot
{
    class Communicator
    {
        private TcpClient clientSocket;

        public Communicator()
        {
            clientSocket = new System.Net.Sockets.TcpClient();
            clientSocket.Connect("127.0.0.1", 8888);
        }
        
        public String SendMessage(String str)
        {
            NetworkStream serverStream = clientSocket.GetStream();
            byte[] outStream = System.Text.Encoding.UTF8.GetBytes(str);
            serverStream.Write(outStream, 0, outStream.Length);
            byte[] inStream = new byte[10025];
            serverStream.Read(inStream, 0, (int)clientSocket.ReceiveBufferSize);
            String response = System.Text.Encoding.UTF8.GetString(inStream);
            return response;
        }

        public void Close()
        {
            clientSocket.Dispose();
        }
    }
}
