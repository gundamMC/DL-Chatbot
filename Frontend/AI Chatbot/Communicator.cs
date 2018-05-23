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

        public Communicator(String ip, int port)
        {
            clientSocket = new System.Net.Sockets.TcpClient();
            clientSocket.Connect(ip, port);
        }
        
        public String SendMessage(String str)
        {
            NetworkStream serverStream = clientSocket.GetStream();
            byte[] outStream = System.Text.Encoding.UTF8.GetBytes(str);
            serverStream.Write(outStream, 0, outStream.Length);
            byte[] inStream = new byte[1024];
            Int32 bytes = serverStream.Read(inStream, 0, (int)inStream.Length);
            String response = System.Text.Encoding.UTF8.GetString(inStream, 0, bytes);
            return response;
        }

        public void Close()
        {
            clientSocket.Dispose();
        }
    }
}
