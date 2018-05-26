using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace AI_Chatbot
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            previousIcon = conversation_tab_image;
            previousImage = new BitmapImage(new Uri("pack://application:,,,/Resources/Icon_Conversation.png"));

            Disable_Conversation();
        }

        private void Title_Bar_MouseDown(object sender, MouseButtonEventArgs e)
        {
            this.DragMove();
        }
        
        private void Input_Field_PreviewMouseDown(object sender, MouseButtonEventArgs e)
        {
            if(input_field.Text == "Input here")
                input_field.Text = "";
        }

        private void input_field_LostFocus(object sender, RoutedEventArgs e)
        {
            if(input_field.Text=="")
                input_field.Text = "Input here";
        }
        

        private void send_button_Click(object sender, RoutedEventArgs e)
        {
            if (input_field.Text == "Input here" || String.IsNullOrWhiteSpace(input_field.Text))
                return;

            // Testing
            ChatBubbleControl input = new ChatBubbleControl() { IsUser = true, Text = input_field.Text };

            conversation_stack_panel.Children.Insert(conversation_stack_panel.Children.Count - 1, input);

            Send(input_field.Text);

            input_field.Text = "Input here";
        }

        Image previousIcon;
        ImageSource previousImage;

        private void conversation_tab_Click(object sender, RoutedEventArgs e)
        {
            Main_TabControl.SelectedIndex = 0;

            previousIcon.Source = previousImage;

            previousIcon = conversation_tab_image;
            previousImage = conversation_tab_image.Source.Clone();

            conversation_tab_image.Source = new BitmapImage(new Uri("pack://application:,,,/Resources/Icon_Conversation_Var.png"));
            
        }

        private void training_tab_Click(object sender, RoutedEventArgs e)
        {
            Main_TabControl.SelectedIndex = 1;

            previousIcon.Source = previousImage;

            previousIcon = training_tab_image;
            previousImage = training_tab_image.Source.Clone();

            training_tab_image.Source = new BitmapImage(new Uri("pack://application:,,,/Resources/Icon_Training_Var.png"));
        }

        private void info_tab_Click(object sender, RoutedEventArgs e)
        {
            Main_TabControl.SelectedIndex = 2;

            previousIcon.Source = previousImage;

            previousIcon = info_tab_image;
            previousImage = info_tab_image.Source.Clone();

            info_tab_image.Source = new BitmapImage(new Uri("pack://application:,,,/Resources/Icon_Info_Var.png"));
        }

        private void input_field_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Return)
            {
                if (input_field.Text == "Input here" || String.IsNullOrWhiteSpace(input_field.Text))
                    return;

                // Testing
                ChatBubbleControl input = new ChatBubbleControl() { IsUser = true, Text = input_field.Text };

                conversation_stack_panel.Children.Insert(conversation_stack_panel.Children.Count - 1, input);

                Send(input_field.Text);

                input_field.Text = "";
                
            }
        }

        private void Send(String message)
        {
            String response = communicator.SendMessage(message);
            ChatBubbleControl bubble = new ChatBubbleControl() { IsUser = false, Text = response};
            conversation_stack_panel.Children.Insert(conversation_stack_panel.Children.Count - 1, bubble);

            Conversation_ScrollViewer.ScrollToBottom();
        }

        private void Enable_Conversation()
        {
            input_field.IsEnabled = true;
            input_field.Text = "Input here";
            send_button.IsEnabled = true;
        }

        private void Disable_Conversation()
        {
            input_field.IsEnabled = false;
            input_field.Text = "Please connect to the socket server";
            send_button.IsEnabled = false;
        }

        Communicator communicator;

        private void Connect_Button_Click(object sender, RoutedEventArgs e)
        {
            if(communicator == null)
            {
                try
                {
                    communicator = new Communicator(IP_Textbox.Text, Int32.Parse(Port_Textbox.Text));
                    Enable_Conversation();
                    Console_Textblock.Text = "Successfully connected to " + IP_Textbox.Text + ":" + Port_Textbox.Text;
                    Connect_Button.Content = "Disconnect";
                }
                catch (Exception ex)
                {
                    Console_Textblock.Text = ex.Message;
                }
            }
            else
            {
                communicator.Close();
                communicator = null;
                Disable_Conversation();
                Console_Textblock.Text = "Disconnected from " + IP_Textbox.Text + ":" + Port_Textbox.Text;
                Connect_Button.Content = "Connect";
            }
        }
            
    }
}
